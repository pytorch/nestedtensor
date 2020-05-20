#pragma once
#include <nestedtensor/csrc/python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

NestedNode<py::object> py_to_nested_node(py::object&& py_obj);

THPNestedTensor nested_tensor(pybind11::sequence list);

} // namespace nested_tensor
} // namespace torch
