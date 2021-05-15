#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;
using namespace c10;

Tensor NestedTensor_expand_as(const Tensor& self_, const Tensor& other) {
  at::Tensor self = self_;
  if (is_nested_tensor_impl(self, other)) {
    TORCH_CHECK(
        get_nested_tensor_impl(self)->nested_dim(),
        get_nested_tensor_impl(other)->nested_dim(),
        "Given NestedTensors need to have same nested dimension.");
    return map_nested_tensor(
        [](at::Tensor s, at::Tensor o) { return at::native::expand_as(s, o); },
        self,
        other);
  }
  TORCH_CHECK(
      !is_nested_tensor_impl(self),
      "Cannot expand a NestedTensor as a Tensor.");
  TORCH_CHECK(
      get_dim(self) <= get_dim(other),
      "Cannot expand to a Tensor of smaller dimension.");
  while (get_dim(self) > 0 && self.size(0) == 1) {
    self = self.squeeze(0);
  }
  return map_nested_tensor(
      [](at::Tensor s, at::Tensor o) { return s.expand_as(o); }, self, other);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "expand_as", NestedTensor_expand_as);
}
} // namespace at
