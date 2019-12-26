#pragma once
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

SizeNode _get_size_structure(py::list py_obj);

TensorNode _get_tensor_structure(py::list py_obj);

static inline THPNestedTensor as_nested_tensor(py::list list) {
  return THPNestedTensor(_ListNestedTensor(_get_tensor_structure(list)));
}

static inline THPNestedTensor _buffer_nested_tensor(
    py::object buffer,
    py::list nested_size) {
  return THPNestedTensor(_BufferNestedTensor(
      py::cast<at::Tensor>(buffer), _get_size_structure(nested_size)));
}

static inline THPNestedTensor _buffer_nested_tensor(
    py::object buffer,
    py::list nested_size,
    py::list nested_stride) {
  return THPNestedTensor(_BufferNestedTensor(
      py::cast<at::Tensor>(buffer),
      _get_size_structure(nested_size),
      _get_size_structure(nested_stride)));
}

} // namespace nested_tensor
} // namespace torch
