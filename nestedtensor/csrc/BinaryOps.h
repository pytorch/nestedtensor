#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/library.h>

namespace at {

inline void check_binary_shape(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self, other)) {
    int64_t self_nested_dim = get_nested_tensor_impl(self)->nested_dim();
    int64_t other_nested_dim = get_nested_tensor_impl(other)->nested_dim();
    TORCH_CHECK(
        self_nested_dim == other_nested_dim,
        "self and other must be of the same nested dimension for NT binary op.");
  } else if (is_nested_tensor_impl(other)) {
    int64_t other_nested_dim = get_nested_tensor_impl(other)->nested_dim();
    TORCH_CHECK(
        self.dim() <= other.dim() - other_nested_dim,
        "tensor dimension of other must match or be greater than dimension of self.");
  } else if (is_nested_tensor_impl(self)) {
    int64_t self_nested_dim = get_nested_tensor_impl(self)->nested_dim();
    TORCH_CHECK(
        other.dim() <= self.dim() - self_nested_dim,
        "tensor dimension of self must match or be greater than dimension of other.");
  } else {
    TORCH_CHECK(false, "check_binary_shape can only be used in NT context.");
  }
}

} // namespace at
