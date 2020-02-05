#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/extension.h>

#include <cstring>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;

c10::optional<IValue> py_obj_to_ivalue(py::object py_obj) {
  auto inferred_type = tryToInferType(py_obj);
  if (!inferred_type.success()) {
    return c10::nullopt;
  }
  auto payload = toIValue(py_obj, inferred_type.type());
  return payload;
}

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  if (size.size() == 0) {
    return 0;
  }
  return size[0] * stride[0];
}

int64_t size_node_memory(SizeNode nested_size, SizeNode nested_stride) {
  auto fn = [](c10::List<int64_t> size,
               c10::List<int64_t> stride,
               int64_t input) { return num_memory(size, stride) + input; };
  return reduce<decltype(fn), int64_t, c10::List<int64_t>, c10::List<int64_t>>(
      nested_size, nested_stride, fn, 0);
}

bool _verify_variables(
    const at::Tensor& first_variable,
    const TensorNode& nested_node) {
  // The attributes must match across all constiuents
  //
  // The NestedTensor's attributes then become that of its
  // constiuents.
  //
  // data must be a list of Tensors or NestedTensors
  //
  // Attributes:
  //     dim()
  //     layout
  //     device
  //     dtype
  //     requires_grad
  //     is_pinned()
  bool valid = true;
  if (nested_node.is_leaf()) {
    const at::Tensor& variable = nested_node.payload();
    // TODO: Add more checks?
    valid = valid && (variable.dim() == first_variable.dim());
    valid = valid && (variable.layout() == first_variable.layout());
    valid = valid && (variable.device() == first_variable.device());
    valid = valid && (variable.dtype() == first_variable.dtype());
    valid =
        valid && (variable.requires_grad() == first_variable.requires_grad());
    // NOTE: This is a very costly check! For now we'll let this to be
    // enabled manually. valid = valid && (variable_.is_pinned() ==
    // first_variable.is_pinned());
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      valid =
          valid && _verify_variables(first_variable, nested_node.children(i));
    }
    for (size_t i = 1; i < nested_node.degree(); i++) {
      valid = valid &&
          (nested_node.children(i).height() ==
           nested_node.children(i - 1).height());
    }
  }
  return valid;
}

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
}

} // namespace nested_tensor
} // namespace torch
