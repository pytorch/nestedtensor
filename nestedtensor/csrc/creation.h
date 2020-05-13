#pragma once
#include <nestedtensor/csrc/nested_tensor.h>
#include <nestedtensor/csrc/py_utils.h>

namespace torch {
namespace nested_tensor {

NestedNode<py::object> py_to_nested_node(py::object&& py_obj);

at::Tensor as_nested_tensor_impl(pybind11::sequence list);

} // namespace nested_tensor
} // namespace torch
