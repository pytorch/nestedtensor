#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_im2col(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  return map_nested_tensor(
      [&](at::Tensor t) {
        return at::im2col(
                   t.unsqueeze(0), kernel_size, dilation, padding, stride)
            .squeeze(0);
      },
      self);
}

Tensor NestedTensor_col2im(
    const Tensor& self,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  return map_nested_tensor(
      [&](at::Tensor t) {
        return at::col2im(
                   t.unsqueeze(0),
                   output_size,
                   kernel_size,
                   dilation,
                   padding,
                   stride)
            .squeeze(0);
      },
      self);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "im2col", NestedTensor_im2col);
  nt_impl(m, "col2im", NestedTensor_col2im);
}

} // namespace at
