#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other)) {
    if (get_is_contiguous(self) && get_is_contiguous(other)) {
      if (get_dim(self) == 3 && get_dim(other) == 2) {
        auto self_opt_sizes = get_opt_sizes(self);
        if (self_opt_sizes[2]) {
          if (*self_opt_sizes[2] == other.size(0)) {
            Tensor self_buffer = get_buffer(self);
            Tensor result_buffer =
                at::matmul(self_buffer.reshape({-1, other.size(0)}), other);
            result_buffer = result_buffer.reshape({-1});
            int64_t other_size_1 = other.size(1);
            EfficientSizeNode new_nested_size =
                get_efficient_nested_size(self).clone();
            EfficientSizeNode new_nested_stride =
                get_efficient_nested_stride(self).clone();
            apply_efficient_size(
                [other_size_1](
                    int64_t* size_ptr,
                    int64_t size_size,
                    int64_t* stride_ptr,
                    int64_t stride_size) {
                  size_ptr[1] = other_size_1;
                  stride_ptr[1] = 1;
                  stride_ptr[0] = other_size_1;
                },
                new_nested_size,
                new_nested_stride);
            return wrap_buffer(
                std::move(result_buffer), new_nested_size, new_nested_stride);
          }
        }
      }
    }
  }
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
    if (get_is_contiguous(input)) {
      if (get_dim(bias) == 1 && get_dim(input) == 3 && get_dim(weight) == 2) {
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
            EfficientSizeNode result_nested_size = map_efficient_size(
                [weight_size_1](int64_t* data_ptr, int64_t size) {
                  data_ptr[1] = weight_size_1;
                },
                get_efficient_nested_size(input));
            EfficientSizeNode input_nested_stride =
                get_efficient_nested_stride(input);
            return wrap_buffer(
                std::move(result_buffer),
                result_nested_size,
                input_nested_stride);
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
