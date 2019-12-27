#pragma once
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

SizeNode _get_size_structure(pybind11::list py_obj);

TensorNode _get_tensor_structure(pybind11::list py_obj);

THPNestedTensor as_nested_tensor(pybind11::list list);

THPNestedTensor _buffer_nested_tensor(
    pybind11::object buffer,
    pybind11::list nested_size);

THPNestedTensor _buffer_nested_tensor(
    pybind11::object buffer,
    pybind11::list nested_size,
    pybind11::list nested_stride);

} // namespace nested_tensor
} // namespace torch
