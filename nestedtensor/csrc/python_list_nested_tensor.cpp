#include <python_list_nested_tensor.h>

namespace torch {
namespace nested_tensor {

static TensorNode _get_tensor_structure(py::list py_obj) {
  // Empty list of Tensors
  if (py_obj.size() == 0) {
    return TensorNode();
  }
  IValue payload = py_obj_to_ivalue(py_obj);
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

THP_ListNestedTensor::THP_ListNestedTensor(py::list list)
    : _data(_ListNestedTensor(_get_tensor_structure(list))) {}
THP_ListNestedTensor::THP_ListNestedTensor(_ListNestedTensor data)
    : _data(data) {}

} // namespace nested_tensor
} // namespace torch
