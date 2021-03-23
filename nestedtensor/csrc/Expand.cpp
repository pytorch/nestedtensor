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

int64_t _tensor_dim(SizeNode nested_size) {
  if (nested_size.is_leaf()) {
    return nested_size.payload().size();
  }
  TORCH_CHECK(nested_size.degree() > 0, "Expected non-zero degree.");
  return _tensor_dim(nested_size.children(0));
}

// bool _sizes_nested_size_equal(
//     SizeNode nested_size,
//     std::vector<int64_t> grad_shape) {
//   if (grad_shape.size() == 0) {
//     return false;
//   }
//   if (nested_size.is_leaf()) {
//     auto payload = nested_size.payload();
//     for (size_t i = 0; i < payload.size(); i++) {
//       if (payload[i] != grad_shape[i]) {
//         return false;
//       }
//     }
//     return true;
//   }
//   if (nested_size.degree() != grad_shape[0]) {
//     return false;
//   }
//   std::vector<int64_t> new_grad_shape;
//   for (size_t i = 1; i < grad_shape.size(); i++) {
//     new_grad_shape.push_back(grad_shape[i]);
//   }
//   for (size_t i = 0; i < nested_size.degree(); i++) {
//     if (!_sizes_nested_size_equal(nested_size.children(i), new_grad_shape)) {
//       return false;
//     }
//   }
//   return true;
// }

bool NestedTensor_sizes_equal(const Tensor& self, IntArrayRef size_other) {
  if (is_nested_tensor_impl(self) && !is_serialized_size_node(size_other)) {
    return false;
  }
  if (!is_nested_tensor_impl(self) && is_serialized_size_node(size_other)) {
    return false;
  }
  if (is_serialized_size_node(size_other)) {
    SizeNode nested_size_other =
        torch::nested_tensor::deserialize_size_node(size_other);
    return nested_size_matches(get_nested_size(self), nested_size_other);
  }
  return self.sizes().equals(size_other);
}

bool NestedTensor_sizes_equal_tensor(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other)) {
    return false;
  }
  if (!is_nested_tensor_impl(self) && is_nested_tensor_impl(other)) {
    return false;
  }
  if (is_nested_tensor_impl(self) && is_nested_tensor_impl(other)) {
    return nested_size_matches(get_nested_size(self), get_nested_size(other));
  }
  return self.sizes().vec() == other.sizes().vec();
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

bool _nested_size_nested_size_expands(SizeNode shape, SizeNode desired) {
  if (shape.is_leaf() && desired.is_leaf()) {
    return at::is_expandable_to(
        IntArrayRef(shape.payload().vec()),
        IntArrayRef(desired.payload().vec()));
  }
  if (shape.is_leaf()) {
    for (size_t i = 0; i < shape.degree(); i++) {
      if (!_nested_size_nested_size_expands(shape, desired.children(i))) {
        return false;
      }
    }
    return true;
  }
  if (desired.is_leaf()) {
    return false;
  }
  if (shape.degree() != desired.degree()) {
    return false;
  }
  for (size_t i = 0; i < shape.degree(); i++) {
    if (!_nested_size_nested_size_expands(
            shape.children(i), desired.children(i))) {
      return false;
    }
  }
  return true;
}

// Can nested_size_other be expanded to match the shape of grad?
// If this is true, a call to sum_to_nt will follow next in autograd/engine.cpp
// to reduce grad down to the shape of nested_size_other.
bool NestedTensor_native_is_expandable_to(
    IntArrayRef metadata_shape, /* shape */
    const Tensor& grad /* desired */) {
  if (is_nested_tensor_impl(grad) && is_serialized_size_node(metadata_shape)) {
    SizeNode nested_size = deserialize_size_node(metadata_shape);
    SizeNode nested_size_desired = get_nested_size(grad);
    return _nested_size_nested_size_expands(nested_size, nested_size_desired);
  }
  if (torch::nested_tensor::is_serialized_size_node(metadata_shape)) {
    SizeNode nested_size =
        torch::nested_tensor::deserialize_size_node(metadata_shape);
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

// Sums `tensor` repeatedly to produce a tensor of shape `shape`.
// Precondition: is_expandable_to(shape, tensor.sizes()) must be true
Tensor NestedTensor_sum_to_size(const Tensor& self, IntArrayRef shape) {
  if (!is_nested_tensor_impl(self) && !is_serialized_size_node(shape)) {
    TORCH_CHECK(
        at::is_expandable_to(shape, self.sizes()),
        "size {",
        shape,
        "} is not expandable to size {",
        self.sizes(),
        "}.");
    return at::sum_to(self, shape);
  }

  int64_t desired_dim;
  if (is_serialized_size_node(shape)) {
    SizeNode nested_size = deserialize_size_node(shape);
    desired_dim = nested_size.height() + _tensor_dim(nested_size);
  } else {
    desired_dim = shape.size();
  }

  TORCH_CHECK(
      desired_dim <= self.dim(),
      "self of lower dimension than desired dimension.");

  if (desired_dim == 0) {
    return self.sum();
  }

  const int64_t leading_dims = self.dim() - desired_dim;
  std::vector<int64_t> reduce_dims;
  for (int64_t i = 0; i < leading_dims; i++) {
    reduce_dims.push_back(i);
  }

  at::Tensor tensor = self;
  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }
  TORCH_CHECK(
      self.dim() == tensor.dim(),
      "internal error: expected self and tensor to be same dim.")

  reduce_dims.clear();
  if (is_nested_tensor_impl(tensor) && is_serialized_size_node(shape)) {
    auto opt_sizes = get_nested_tensor_impl(tensor)->opt_sizes();
    SizeNode nested_size = deserialize_size_node(shape);
    auto opt_sizes_desired = construct_size(nested_size);
    for (int64_t i = leading_dims; i < static_cast<int64_t>(self.dim()); ++i) {
      if (opt_sizes_desired[i - leading_dims] &&
          (*opt_sizes_desired[i - leading_dims]) == 1 &&
          !(opt_sizes[i] && (*opt_sizes[i]) == 1)) {
        reduce_dims.push_back(i);
      }
    }
  }
  if (!is_nested_tensor_impl(tensor) && is_serialized_size_node(shape)) {
    auto sizes = tensor.sizes();
    SizeNode nested_size = deserialize_size_node(shape);
    auto opt_sizes_desired = construct_size(nested_size);
    for (int64_t i = leading_dims; i < static_cast<int64_t>(self.dim()); ++i) {
      if (opt_sizes_desired[i - leading_dims] &&
          (*opt_sizes_desired[i - leading_dims]) == 1 && !(sizes[i] == 1)) {
        reduce_dims.push_back(i);
      }
    }
  }
  if (is_nested_tensor_impl(tensor) && !is_serialized_size_node(shape)) {
    auto opt_sizes = get_nested_tensor_impl(tensor)->opt_sizes();
    for (int64_t i = leading_dims; i < static_cast<int64_t>(self.dim()); ++i) {
      if (shape[i - leading_dims] == 1 &&
          !(opt_sizes[i] && (*opt_sizes[i]) == 1)) {
        reduce_dims.push_back(i);
      }
    }
  }
  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }
  if (is_nested_tensor_impl(tensor) && is_serialized_size_node(shape)) {
    SizeNode desired_nested_size = deserialize_size_node(shape);
    TORCH_CHECK(
        get_nested_size(tensor).height() == desired_nested_size.height(),
        "internal error: expected result tensor height and desired shape to match.");
    return wrap_tensor_node(map(
        [](at::Tensor t, c10::List<int64_t> s) {
          return t.sum_to_size(IntArrayRef(s.vec()));
        },
        get_nested_tensor_structure(tensor),
        desired_nested_size));
  }
  if (!is_nested_tensor_impl(tensor) && is_serialized_size_node(shape)) {
    SizeNode desired_nested_size = deserialize_size_node(shape);
    return wrap_buffer(tensor.reshape({-1}), desired_nested_size);
  }
  if (is_nested_tensor_impl(tensor) && !is_serialized_size_node(shape)) {
    tensor = NestedTensor_to_tensor(tensor, c10::nullopt);
  }
  return leading_dims > 0 ? tensor.view(shape) : tensor;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "expand_as", NestedTensor_expand_as);
}
TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "expand_nt", NestedTensor_expand_nt);
  nt_impl(m, "native_is_expandable_to", NestedTensor_native_is_expandable_to);
  nt_impl(m, "sizes_equal", NestedTensor_sizes_equal);
  nt_impl(m, "sizes_equal.tensor", NestedTensor_sizes_equal_tensor);
  nt_impl(m, "sum_to_size", NestedTensor_sum_to_size);
}
TORCH_LIBRARY_IMPL(aten, AutogradCPU, m) {
  nt_impl(m, "expand_nt", NestedTensor_expand_nt);
  nt_impl(m, "native_is_expandable_to", NestedTensor_native_is_expandable_to);
  nt_impl(m, "sizes_equal", NestedTensor_sizes_equal);
  nt_impl(m, "sizes_equal.tensor", NestedTensor_sizes_equal_tensor);
  nt_impl(m, "sum_to_size", NestedTensor_sum_to_size);
}
TORCH_LIBRARY_IMPL(aten, AutogradCUDA, m) {
  nt_impl(m, "expand_nt", NestedTensor_expand_nt);
  nt_impl(m, "native_is_expandable_to", NestedTensor_native_is_expandable_to);
  nt_impl(m, "sizes_equal", NestedTensor_sizes_equal);
  nt_impl(m, "sizes_equal.tensor", NestedTensor_sizes_equal_tensor);
  nt_impl(m, "sum_to_size", NestedTensor_sum_to_size);
}
} // namespace at
