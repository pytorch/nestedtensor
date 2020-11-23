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

bool NestedTensor_sizes_equal_nt_other(
    const Tensor& self,
    IntArrayRef nested_size_other) {
  // TODO: This does nothing right now
  SizeNode nested_size =
      torch::nested_tensor::deserialize_size_node(nested_size_other);
  if (is_nested_tensor_impl(self)) {
    return false;
    // return torch::nested_tensor::shape_matches(
    //     get_nested_tensor_impl(self)->nested_size(), nested_size);
  }
  return false;
}

int64_t _tensor_dim(SizeNode nested_size) {
  if (nested_size.is_leaf()) {
    return nested_size.payload().size();
  }
  TORCH_CHECK(nested_size.degree() > 0, "Expected non-zero degree.");
  return _tensor_dim(nested_size.children(0));
}

bool _sizes_nested_size_equal(
    SizeNode nested_size,
    std::vector<int64_t> grad_shape) {
  if (grad_shape.size() == 0) {
    return false;
  }
  if (nested_size.is_leaf()) {
    auto payload = nested_size.payload();
    for (size_t i = 0; i < payload.size(); i++) {
      if (payload[i] != grad_shape[i]) {
        return false;
      }
    }
    return true;
  }
  if (nested_size.degree() != grad_shape[0]) {
    return false;
  }
  std::vector<int64_t> new_grad_shape;
  for (size_t i = 1; i < grad_shape.size(); i++) {
    new_grad_shape.push_back(grad_shape[i]);
  }
  for (size_t i = 0; i < nested_size.degree(); i++) {
    if (!_sizes_nested_size_equal(nested_size.children(i), new_grad_shape)) {
      return false;
    }
  }
  return true;
}

bool NestedTensor_sizes_equal(const Tensor& self, IntArrayRef size_other) {
  if (self.dim() != size_other.size()) {
    return false;
  }
  return _sizes_nested_size_equal(get_nested_size(self), size_other.vec());
}

bool _sizes_nested_size_expands(
    SizeNode nested_size,
    std::vector<int64_t> grad_shape) {
  if (grad_shape.size() == 0) {
    return false;
  }
  if (nested_size.is_leaf()) {
    return is_expandable_to(
        IntArrayRef(nested_size.payload().vec()), IntArrayRef(grad_shape));
  }
  if (nested_size.degree() != grad_shape[0] && nested_size.degree() != 1) {
    return false;
  }
  std::vector<int64_t> new_grad_shape;
  for (size_t i = 1; i < grad_shape.size(); i++) {
    new_grad_shape.push_back(grad_shape[i]);
  }
  for (size_t i = 0; i < nested_size.degree(); i++) {
    if (!_sizes_nested_size_expands(nested_size.children(i), new_grad_shape)) {
      return false;
    }
  }
  return true;
}

// Can nested_size_other be expanded to match the shape of grad?
// If this is true, a call to sum_to_nt will follow next in autograd/engine.cpp
// to reduce grad down to the shape of nested_size_other.
bool NestedTensor_native_is_expandable_to_nt_other(
    IntArrayRef nested_size_other /* shape */,
    const Tensor& grad /* desired */) {
  SizeNode nested_size =
      torch::nested_tensor::deserialize_size_node(nested_size_other);
  if (is_nested_tensor_impl(grad)) {
    return torch::nested_tensor::shape_matches(
        get_nested_size(grad), nested_size);
  }
  int64_t nested_size_dim = nested_size.height() + _tensor_dim(nested_size);
  if (nested_size_dim > grad.dim()) {
    return false;
  }
  std::vector<int64_t> grad_shape = grad.sizes().vec();
  if (nested_size_dim < grad.dim()) {
    std::vector<int64_t> new_grad_shape;
    for (int64_t i = grad.dim() - nested_size_dim; i < grad.dim(); i++) {
      new_grad_shape.push_back(grad_shape[i]);
    }
    grad_shape = new_grad_shape;
  }
  return _sizes_nested_size_expands(nested_size, grad_shape);
}

bool NestedTensor_native_is_expandable_to(
    IntArrayRef metadata_shape, /* shape */
    const Tensor& grad /* desired */) {
  if (torch::nested_tensor::is_serialized_size_node(metadata_shape)) {
    return NestedTensor_native_is_expandable_to_nt_other(metadata_shape, grad);
  }
  if (is_nested_tensor_impl(grad)) {
    auto fn = [&metadata_shape](at::Tensor leaf, bool input) {
      return input && at::is_expandable_to(metadata_shape, leaf.sizes());
    };
    return reduce<decltype(fn), bool, at::Tensor>(
        get_nested_tensor_structure(grad), fn, true);
  }
  return at::is_expandable_to(metadata_shape, grad.sizes());
}

Tensor NestedTensor_expand_nt(
    const Tensor& self,
    const Tensor& nested_size_tensor,
    bool implicit) {
  TORCH_CHECK(!is_nested_tensor_impl(self), "Expected regular tensor as self.");
  SizeNode nested_size =
      torch::nested_tensor::deserialize_size_node(nested_size_tensor);
  TORCH_CHECK(
      self.dim() <= _tensor_dim(nested_size),
      "self dim can't exceed nested_size tensor dim.");
  // TODO: This doesn't support NT broadcasting of leading dimensions
  return wrap_tensor_node(map(
      [](at::Tensor self, c10::List<int64_t> size) {
        return at::native::expand(self, IntArrayRef(size.vec()));
      },
      get_nested_tensor_structure(self),
      nested_size));
}

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
      self.dim() <= other.dim(),
      "Cannot expand to a Tensor of smaller dimension.");
  while (self.dim() > 0 && self.size(0) == 1) {
    self = self.squeeze(0);
  }
  return map_nested_tensor(
      [](at::Tensor s, at::Tensor o) { return s.expand_as(o); }, self, other);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "expand_as", NestedTensor_expand_as);
  nt_impl(m, "sizes_equal", NestedTensor_sizes_equal);
}
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  nt_impl(m, "expand_nt", NestedTensor_expand_nt);
  nt_impl(m, "sizes_equal_nt_other", NestedTensor_sizes_equal_nt_other);
  nt_impl(m, "native_is_expandable_to", NestedTensor_native_is_expandable_to);
}
} // namespace at
