#include <creation.h>
#include <nested_node.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

// TODO: Support for THPNestedTensor as part of given data.

TensorNode _get_tensor_structure(py::sequence py_obj) {
  // Empty list of Tensors
  if (py_obj.size() == 0) {
    std::cout << "size 0 " << std::endl;
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
      py::sequence py_obj_i = py::sequence(py_obj[i]);
      result.push_back(_get_tensor_structure(py_obj_i));
    }
    return TensorNode(result);
  }
}

void _make_tensors(py::sequence py_obj, std::vector<at::Tensor>& tensors) {
  // Empty list of Tensors
  c10::IValue payload = py_obj_to_ivalue(py_obj);
  if (payload.isTensorList()) {
    // List of Tensors
    c10::List<at::Tensor> tensor_list = payload.toTensorList();
    for (size_t i = 0; i < tensor_list.size(); i++) {
      tensors.push_back(tensor_list.get(i).reshape({-1}));
    }
  } else {
    // List of lists of Tensors
    for (size_t i = 0; i < py_obj.size(); i++) {
      py::sequence py_obj_i = py::sequence(py_obj[i]);
      _make_tensors(py_obj_i, tensors);
    }
  }
}

THPNestedTensor as_nested_tensor(py::list list) {
  return THPNestedTensor(_ListNestedTensor(_get_tensor_structure(list)));
}

// TODO: Support empty list.
// TODO: Support THPNestedTensor entries
THPNestedTensor nested_tensor(py::sequence list) {
  // std::cout << "list: " << list << std::endl;
  std::cout << "1" << std::endl;
  TensorNode structure = _get_tensor_structure(list);
  std::cout << "structure.degree(): " << structure.degree() << std::endl;
  std::cout << "structure.size(): " << structure.size() << std::endl;
  std::vector<at::Tensor> tensors;
  _make_tensors(list, tensors);
  at::Tensor buffer;
  if (list.size() == 0) {
    buffer = torch::ones({});
  } else {
    buffer = at::cat(tensors, 0);
  }
  std::cout << "2" << std::endl;
  SizeNode nested_size = map<at::Tensor, c10::List<int64_t>>(
      structure, [](at::Tensor tensor) -> c10::List<int64_t> {
        return c10::List<int64_t>(tensor.sizes());
      });
  std::cout << "3" << std::endl;
  std::cout << "nested_size.degree(): " << nested_size.degree() << std::endl;
  std::cout << "nested_size.size(): " << nested_size.size() << std::endl;
  return THPNestedTensor(_BufferNestedTensor(buffer, nested_size));
}

} // namespace nested_tensor
} // namespace torch
