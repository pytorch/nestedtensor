#include <python_buffer_nested_tensor.h>

namespace torch {
namespace nested_tensor {

static SizeNode _get_size_structure(py::list py_obj) {
  // Empty list of lists
  if (py_obj.size() == 0) {
    return SizeNode();
  }

  // List of empty lists
  py::list py_obj_0 = py_obj[0];
  if (py_obj_0.size() == 0) {
    c10::List<c10::List<int64_t>> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      result.push_back(c10::List<int64_t>());
    }
    return SizeNode(result);
  }

  // List of lists of numbers
  InferredType inferred_type = tryToInferType(py_obj[0]);
  if (inferred_type.success() && py_obj_to_ivalue(py_obj[0]).isIntList()) {
    c10::List<c10::List<int64_t>> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      result.push_back(py_obj_to_ivalue(py_obj[i]).toIntList());
    }
    return SizeNode(result);
  }

  // List of lists of lists...
  std::vector<SizeNode> result;
  for (size_t i = 0; i < py_obj.size(); i++) {
    py::list py_obj_i = py_obj[i];
    result.emplace_back(_get_size_structure(py_obj_i));
  }
  return SizeNode(result);
}

THP_BufferNestedTensor::THP_BufferNestedTensor(_BufferNestedTensor data)
    : _data(data) {}
THP_BufferNestedTensor::THP_BufferNestedTensor(
    py::object buffer,
    py::list nested_size)
    : _data(_BufferNestedTensor(
          py::cast<at::Tensor>(buffer),
          _get_size_structure(nested_size))) {}
THP_BufferNestedTensor::THP_BufferNestedTensor(
    py::object buffer,
    py::list nested_size,
    py::list nested_stride)
    : _data(_BufferNestedTensor(
          py::cast<at::Tensor>(buffer),
          _get_size_structure(nested_size),
          _get_size_structure(nested_stride))) {}

} // namespace nested_tensor
} // namespace torch
