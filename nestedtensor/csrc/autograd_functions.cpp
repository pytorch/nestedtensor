#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  return map_nested_tensor(
      [&](const at::Tensor t) { return at::dropout(t, p, train); }, input);
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
  int64_t n_input = *opt_sizes[1];
  if (running_mean) {
    check_dims_match_num_input_features(
        "running_mean", n_input, running_mean->numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var) {
    check_dims_match_num_input_features(
        "running_var", n_input, running_var->numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight) {
    check_dims_match_num_input_features(
        "weight", n_input, weight->numel());
  }
  if (bias) {
    check_dims_match_num_input_features("bias", n_input, bias->numel());
  }

  auto scalar_shape = make_scalar_shape(input.dim(), n_input);

  at::Tensor mean;
  at::Tensor invstd;
  at::Tensor save_mean;
  at::Tensor save_invstd;

  if (training) {
    auto reduce_dims = make_reduce_dims(input.dim());
    save_mean = at::mean(input, IntArrayRef(reduce_dims));

    save_invstd =
        1 / at::sqrt(at::var(input, IntArrayRef(reduce_dims), false) + eps);

    if (running_mean) {
      at::Tensor running_mean_(running_mean->getIntrusivePtr());
      running_mean_ = running_mean_.detach();
      running_mean_.copy_(
          momentum * save_mean + (1 - momentum) * running_mean_);
    }

    if (running_var) {
      Tensor unbiased_var = at::var(input, IntArrayRef(reduce_dims));
      at::Tensor running_var_(running_var->getIntrusivePtr());
      running_var_ = running_var_.detach();
      running_var_.copy_(
          momentum * unbiased_var + (1 - momentum) * running_var_);
    }

    mean = save_mean;
    invstd = save_invstd;
  } else {
    mean = *running_mean;
    invstd = 1 / at::sqrt(*running_var + eps);
  }

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
