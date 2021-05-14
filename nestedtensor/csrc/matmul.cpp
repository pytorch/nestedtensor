#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  return map_nested_tensor(
      [](at::Tensor self, at::Tensor other) { return at::matmul(self, other); },
      self,
      other);
}

Tensor NestedTensor_addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const c10::Scalar& alpha,
    const c10::Scalar& beta) {
  if (!is_nested_tensor_impl(bias) && is_nested_tensor_impl(input) &&
      !is_nested_tensor_impl(weight)) {
    if (bias.is_contiguous() && input.is_contiguous() &&
        weight.is_contiguous()) {
      if (bias.dim() == 1 && input.dim() == 3 && weight.dim() == 2) {
        auto input_opt_sizes = get_opt_sizes(input);
        if (input_opt_sizes[2]) {
          if (*input_opt_sizes[2] == weight.size(1)) {
            Tensor input_buffer = get_buffer(input);
            Tensor result_buffer =
                at::addmm(
                    bias,
                    input_buffer.reshape({-1, weight.size(1)}),
                    weight,
                    alpha,
                    beta)
                    .reshape({-1});
            int64_t weight_size_1 = weight.size(1);
            EfficientSizeNode input_nested_size =
                get_efficient_nested_size(input);
            apply(
                [weight_size_1](int64_t* data_ptr, int64_t size) {
                  data_ptr[1] = weight_size_1;
                },
                input_nested_size);
            EfficientSizeNode input_nested_stride =
                get_efficient_nested_stride(input);
            return wrap_buffer(std::move(result_buffer), input_nested_size, input_nested_stride);
          }
        }
      }
    }
  }
  return map_nested_tensor(
      [&alpha, &beta](at::Tensor bias, at::Tensor input, at::Tensor weight) {
        return at::addmm(bias, input, weight, alpha, beta);
      },
      bias,
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "addmm", NestedTensor_addmm);
  nt_impl(m, "matmul", NestedTensor_matmul);
}
} // namespace at
