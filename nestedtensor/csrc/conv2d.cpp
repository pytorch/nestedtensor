#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (bias) {
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
        return at::conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
        // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
      },
      input,
      weight,
      *bias);
  }
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight) {
        return at::conv2d(input.unsqueeze(0), weight, c10::nullopt, stride, padding, dilation, groups).squeeze(0);
        // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
      },
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
