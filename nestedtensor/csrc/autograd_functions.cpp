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

  at::Tensor mean = *running_mean;
  at::Tensor var = *running_var;
#ifdef WITH_CUDA
  if (weight &&
      bias &&
      (is_nested_tensor_impl(input)) &&
      (!is_nested_tensor_impl(mean)) &&
      (!is_nested_tensor_impl(var)) &&
      (!is_nested_tensor_impl(*bias)) &&
      (!is_nested_tensor_impl(*weight)) &&
      (input.dtype()   == torch::kHalf) &&
      (mean.dtype()    == torch::kHalf) &&
      (var.dtype()     == torch::kHalf) &&
      (bias->dtype()   == torch::kHalf) &&
      (weight->dtype() == torch::kHalf) &&
      get_is_cuda(input)
  )
  {
    // Custom CUDA Half implementation.
    mean = mean.contiguous();
    Tensor bias_cont = (*bias).contiguous();
    Tensor weight_cont = (*weight).contiguous();
    Tensor running_var_cont = (*running_var).contiguous();

    c10::Half* mean_ptr = mean.data_ptr<c10::Half>();
    c10::Half* bias_ptr = bias_cont.data_ptr<c10::Half>();
    c10::Half* weight_ptr = weight_cont.data_ptr<c10::Half>();
    c10::Half* running_var_ptr = running_var_cont.data_ptr<c10::Half>();

    if (get_is_contiguous(input, c10::MemoryFormat::ChannelsLast)) {
      Tensor input_buffer = get_buffer(input);
      int64_t num_channel = weight_cont.size(0);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      nested_tensor::cuda::batchnorm_inference_channels_last_kernelLauncher(
          input_buffer.data_ptr<c10::Half>(),
          mean_ptr,
          running_var_ptr,
          c10::Half((float)(eps)),
          weight_ptr,
          bias_ptr,
          input_buffer.data_ptr<c10::Half>(),
          num_channel,
          input_buffer.numel(),
          defaultStream);
      input_buffer = input_buffer.view(-1);
      return wrap_buffer(std::move(input_buffer), get_efficient_nested_size(input), get_efficient_nested_stride(input));
    }
  
    Tensor output = input;
    output = NestedTensor_contiguous(output);
    Tensor input_buffer = get_buffer(output);
    // Tensor output_buffer = input_buffer.clone();
  
    auto self_opt_sizes = get_opt_sizes(input);
  
    Tensor nt_sizes_ =
        get_efficient_nested_size(input).sizes(); // .to(torch::kInt32);
    Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1);
    Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1);
    Tensor nt_sizes_all = nt_sizes_1 * nt_sizes_2;
    int64_t* nt_sizes_all_ptr = nt_sizes_all.data_ptr<int64_t>();
    at::Tensor numbers_t = at::empty({1 + (nt_sizes_all.size(0) * *self_opt_sizes[1])}, torch::kInt64);
    int64_t* numbers_t_ptr = numbers_t.data_ptr<int64_t>();
    numbers_t_ptr[0] = 0;
    int64_t index = 1;
    for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
      for (int64_t j = 0; j < *self_opt_sizes[1]; j++) {
        numbers_t_ptr[index] = (numbers_t_ptr[index - 1] + nt_sizes_all_ptr[i]);
        index++;
      }
    }
    Tensor nt_sizes = numbers_t.to(at::Device(kCUDA), torch::kInt32, true, true);
  
    at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
    nested_tensor::cuda::batchnorm_inference_kernelLauncher(
        input_buffer.data_ptr<c10::Half>(),
        mean_ptr,
        running_var_ptr,
        c10::Half((float)(eps)),
        weight_ptr,
        bias_ptr,
        input_buffer.data_ptr<c10::Half>(),
        // output_buffer.data_ptr<c10::Half>(),
        (int)(*self_opt_sizes[0]),
        (int)(weight_cont.size(0)),
        (int)(*self_opt_sizes[0] *
              *self_opt_sizes[1] *
              *self_opt_sizes[2] *
              *self_opt_sizes[3]),
        nt_sizes.data_ptr<int>(),
        defaultStream
        );
    return wrap_buffer(std::move(input_buffer), get_efficient_nested_size(output), get_efficient_nested_stride(output));
  }
#endif
  auto scalar_shape = make_scalar_shape(get_dim(input), n_input);

  at::Tensor invstd = 1 / at::sqrt(*running_var + eps);

  Tensor output = input;
  output = output - mean.reshape(IntArrayRef(scalar_shape));
  output = output * invstd.reshape(IntArrayRef(scalar_shape));

  if (weight) {
    output = output * weight->reshape(IntArrayRef(scalar_shape));
  }
  if (bias) {
    output = output + bias->reshape(IntArrayRef(scalar_shape));
  }
  return output;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  // nt_impl(m, "upsample_bilinear2d", NestedTensor_upsample_bilinear2d);
  nt_impl(m, "clone", NestedTensor_clone);
  nt_impl(m, "dropout", NestedTensor_dropout);
  nt_impl(m, "batch_norm", NestedTensor_batch_norm);
}

} // namespace at
