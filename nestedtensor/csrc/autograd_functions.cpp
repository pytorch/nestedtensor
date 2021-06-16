#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/add.h>
#include <c10/util/Half.h>
#endif

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  if (train) {
    return map_nested_tensor(
        [&](const at::Tensor t) { return at::dropout(t, p, train); }, input);
  }
  return input;
}

Tensor NestedTensor_upsample_bilinear2d(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  return map_nested_tensor(
      [&](at::Tensor t) {
        return at::upsample_bilinear2d(
                   t.unsqueeze(0),
                   output_size,
                   align_corners,
                   scales_h,
                   scales_w)
            .squeeze(0);
      },
      input);
}

Tensor NestedTensor_clone(
    const Tensor& src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return map_nested_tensor(
      [&optional_memory_format](Tensor a) {
        return at::clone(a, optional_memory_format);
      },
      src);
}

void check_dims_match_num_input_features(
    const char* arg_name,
    int64_t expected,
    int64_t actual) {
  TORCH_CHECK(
      actual == expected,
      arg_name,
      " should contain ",
      expected,
      " elements not ",
      actual);
}

std::vector<int64_t> make_reduce_dims(int64_t input_dim) {
  std::vector<int64_t> result;
  result.push_back(0);
  for (int64_t i = 2; i < input_dim; i++) {
    result.push_back(i);
  }
  return result;
}

std::vector<int64_t> make_scalar_shape(int64_t input_dim, int64_t n_input) {
  std::vector<int64_t> result;
  result.push_back(1);
  result.push_back(n_input);
  for (int64_t i = 2; i < input_dim; i++) {
    result.push_back(1);
  }
  return result;
}

Tensor NestedTensor_batch_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight /* optional */,
    const c10::optional<Tensor>& bias /* optional */,
    const c10::optional<Tensor>& running_mean /* optional */,
    const c10::optional<Tensor>& running_var /* optional */,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  auto opt_sizes = get_nested_tensor_impl(input)->opt_sizes();
  TORCH_CHECK(opt_sizes[1], "batch norm requires regular second dimension.");
  TORCH_CHECK(!training, "batch norm does not support training.");
  int64_t n_input = *opt_sizes[1];
  TORCH_CHECK(running_mean, "running_mean must be defined in evaluation mode");
  TORCH_CHECK(running_var, "running_var must be defined in evaluation mode");
  if (weight) {
    check_dims_match_num_input_features("weight", n_input, get_numel(*weight));
  }
  if (bias) {
    check_dims_match_num_input_features("bias", n_input, get_numel(*bias));
  }

  auto scalar_shape = make_scalar_shape(get_dim(input), n_input);
  // std::cout << "IntArrayRef(scalar_shape): " << IntArrayRef(scalar_shape) << std::endl;
  // std::cout << "weight.has_value(): " << weight.has_value() << std::endl;
  // std::cout << "bias.has_value(): " << bias.has_value() << std::endl;
  // std::cout << "is_nested_tensor_impl(running_mean): " << is_nested_tensor_impl(*running_mean) << std::endl;
  // std::cout << "is_nested_tensor_impl(running_var): " << is_nested_tensor_impl(*running_var) << std::endl;
  // std::cout << "is_nested_tensor_impl(bias): " << is_nested_tensor_impl(*bias) << std::endl;
  // std::cout << "is_nested_tensor_impl(weight): " << is_nested_tensor_impl(*weight) << std::endl;
  // std::cout << "(running_mean): " << (*running_mean).sizes() << std::endl;
  // std::cout << "(running_var): " << (*running_var).sizes() << std::endl;
  // std::cout << "(bias): " << (*bias).sizes() << std::endl;
  // std::cout << "(weight): " << (*weight).sizes() << std::endl;
  // map([](std::vector<int64_t> size) {
  //     std::cout << "IntArrayRef(size): " << IntArrayRef(size) << std::endl;
  //     return size;
  //     }, get_nested_size(input));

  at::Tensor mean;
  at::Tensor invstd;


  mean = *running_mean;
  invstd = 1 / at::sqrt(*running_var + eps);
  // mean = mean.to(torch::kHalf);
  // invstd = invstd.to(torch::kHalf);


  // Custom CUDA Half implementation.
  mean = mean.contiguous();
  invstd = invstd.contiguous();
  Tensor bias_cont = (*bias).contiguous();
  Tensor weight_cont = (*weight).contiguous();

  Tensor output = input;
  output = NestedTensor_contiguous(output);
  Tensor input_buffer = get_buffer(output);
  Tensor output_buffer = input_buffer.clone();

  auto self_opt_sizes = get_opt_sizes(input);

  Tensor nt_sizes_ =
      get_efficient_nested_size(input).sizes().to(torch::kInt32);
  Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1);
  Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1);
  Tensor nt_sizes_all = nt_sizes_1 * nt_sizes_2;
  std::vector<int> numbers;
  for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
    for (int64_t j = 0; j < *self_opt_sizes[1]; j++) {
      numbers.push_back(nt_sizes_all[i].item<int>());
    }
  }
  at::Tensor numbers_t = torch::tensor(numbers).to(torch::kInt32);
  Tensor nt_sizes_cumsum =
      at::native::cumsum(numbers_t, 0).to(torch::kInt32).reshape({-1});
  TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor metadata of unexpected dimension.")
  Tensor nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes_cumsum});
  nt_sizes = nt_sizes.to(torch::kCUDA);

  c10::Half* mean_ptr = mean.data_ptr<c10::Half>();
  c10::Half* invstd_ptr = invstd.data_ptr<c10::Half>();
  c10::Half* bias_ptr = bias_cont.data_ptr<c10::Half>();
  c10::Half* weight_ptr = weight_cont.data_ptr<c10::Half>();

  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  nested_tensor::cuda::batchnorm_inference_kernelLauncher(
      input_buffer.data_ptr<c10::Half>(),
      mean_ptr,
      invstd_ptr,
      weight_ptr,
      bias_ptr,
      output_buffer.data_ptr<c10::Half>(),
      (int)(*self_opt_sizes[0] * *self_opt_sizes[1]),
      (int)(*self_opt_sizes[0]),
      nt_sizes.data_ptr<int>(),
      defaultStream
      );
  return wrap_buffer(std::move(output_buffer), get_efficient_nested_size(output), get_efficient_nested_stride(output));

  // Tensor input_buffer = get_buffer(input);
  // Tensor output_buffer 

  // Tensor output = input;
  // output = output - mean.reshape(IntArrayRef(scalar_shape));
  // output = output * invstd.reshape(IntArrayRef(scalar_shape));

  // if (weight) {
  //   output = output * weight->reshape(IntArrayRef(scalar_shape));
  // }
  // if (bias) {
  //   output = output + bias->reshape(IntArrayRef(scalar_shape));
  // }
  // return output;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  // nt_impl(m, "upsample_bilinear2d", NestedTensor_upsample_bilinear2d);
  nt_impl(m, "clone", NestedTensor_clone);
  nt_impl(m, "dropout", NestedTensor_dropout);
  nt_impl(m, "batch_norm", NestedTensor_batch_norm);
}

} // namespace at
