#include <creation.h>
#include <nested_node.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

SizeNode _get_size_structure(py::list py_obj) {
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
  torch::jit::InferredType inferred_type = torch::jit::tryToInferType(py_obj[0]);
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

TensorNode _get_tensor_structure(py::list py_obj) {
  // Empty list of Tensors
  if (py_obj.size() == 0) {
    return TensorNode();
  }
  c10::IValue payload = py_obj_to_ivalue(py_obj);
  if (payload.isTensorList()) {
    // List of Tensors
    return TensorNode(payload.toTensorList());
  } else {
    // List of lists of Tensors
    std::vector<TensorNode> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      py::list py_obj_i = py::list(py_obj[i]);
      result.push_back(_get_tensor_structure(py_obj_i));
    }
    return TensorNode(result);
  }
}

THPNestedTensor as_nested_tensor(py::list list) {
  return THPNestedTensor(_ListNestedTensor(_get_tensor_structure(list)));
}

THPNestedTensor _buffer_nested_tensor(
    py::object buffer,
    py::list nested_size) {
  return THPNestedTensor(_BufferNestedTensor(
      py::cast<at::Tensor>(buffer), _get_size_structure(nested_size)));
}

THPNestedTensor _buffer_nested_tensor(
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
