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
        get_dim(self) <= get_dim(other) - other_nested_dim,
        "tensor dimension of other must match or be greater than dimension of self.");
  } else if (is_nested_tensor_impl(self)) {
    int64_t self_nested_dim = get_nested_tensor_impl(self)->nested_dim();
    TORCH_CHECK(
        get_dim(other) <= get_dim(self) - self_nested_dim,
        "tensor dimension of self must match or be greater than dimension of other.");
  } else {
    TORCH_CHECK(false, "check_binary_shape can only be used in NT context.");
  }
}

inline std::tuple<at::Tensor, at::Tensor> _expand_other_as(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self, other)) {
    int64_t self_nested_dim = get_nested_tensor_impl(self)->nested_dim();
    int64_t other_nested_dim = get_nested_tensor_impl(other)->nested_dim();
    TORCH_CHECK(
        self_nested_dim == other_nested_dim,
        "self and other must be of the same nested dimension for NT binary op.");
    return std::make_tuple(self, other);
  }
  if (is_nested_tensor_impl(other)) {
    auto result = _expand_other_as(other, self);
    return std::make_tuple(std::get<1>(result), std::get<0>(result));
  }
  // self is a NestedTensor, other is a Tensor
  TORCH_CHECK(
      is_nested_tensor_impl(self),
      "_expand_other_as can only be used in NT context.");
  if (get_dim(other) >= get_dim(self)) {
    at::Tensor other_nt = NestedTensor_to_nested_tensor(other, get_nested_dim(self));
    return std::make_tuple(self, other_nt);
  }
  int64_t self_nested_dim = get_nested_tensor_impl(self)->nested_dim();
  if (get_dim(other) + self_nested_dim >= get_dim(self)) {
    at::Tensor other_ = other;
    for (int64_t i = 0; i < self_nested_dim; i++) {
      if (other.size(0) == 1) {
        other_ = other_.squeeze(0);
      } else {
        TORCH_CHECK(
            "Expected singleton at dimension ",
            std::to_string(i),
            " but got ",
            std::to_string(other.size(0)));
      }
    }
    return std::make_tuple(self, other_);
  }
  return std::make_tuple(self, other);
}

} // namespace at
