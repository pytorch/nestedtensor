#pragma once
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <nestedtensor/csrc/utils/nested_node.h>

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;

template <typename A, typename B = py::object>
B wrap_nested_node(NestedNode<A> nested_node) {
  if (nested_node.is_leaf()) {
    return B(torch::jit::toPyObject(nested_node.payload()));
  } else {
    std::vector<B> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.push_back(wrap_nested_node(nested_node.children(i)));
    }
    return B(py::cast(result));
  }
}

template <typename T, typename F>
std::string NestedNode___str__(
    const NestedNode<T>& nested_node,
    const std::string& name,
    F payload_to_str,
    const std::string& tabs = "") {
  std::stringstream result;
  if (nested_node.is_leaf()) {
    result << payload_to_str(nested_node.payload(), tabs);
  } else {
    auto tabs_ = tabs + "\t";
    result << tabs;
    result << name;
    result << "([";
    result << std::endl;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      if (i > 0) {
        result << ",";
        result << std::endl;
      }
      result << NestedNode___str__<T, F>(
          nested_node.children(i), name, payload_to_str, tabs_);
    }
    result << std::endl;
    result << tabs;
    result << "])";
  }
  return result.str();
}

} // namespace nested_tensor
} // namespace torch
