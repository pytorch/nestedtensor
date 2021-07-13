#pragma once
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/py_utils.h>

namespace torch {
namespace nested_tensor {

NestedNode<py::object> py_to_nested_node(py::object&& py_obj);

at::Tensor nested_tensor_impl(
    pybind11::sequence list,
    pybind11::object dtype,
    pybind11::object device,
    bool requires_grad,
    bool pin_memory,
    bool channels_last);

} // namespace nested_tensor
} // namespace torch
