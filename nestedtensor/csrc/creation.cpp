#include <creation.h>
#include <nested_node.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

// TODO: Support for THPNestedTensor as part of given data.
// TODO: Generalize this for SizeNodes
c10::optional<c10::List<at::Tensor>> to_tensor_sequence(
    const py::sequence& py_obj) {
  bool result = true;
  for (size_t i = 0; i < py_obj.size(); i++) {
    auto payload = py_obj_to_ivalue(py_obj[i]);
    if (!payload) {
      result = false;
      break;
    }
    result = result && (*payload).isTensor();
  }
  if (!result) {
    return c10::nullopt;
  }
  c10::List<at::Tensor> tensors;
  tensors.resize(py_obj.size());
  for (size_t i = 0; i < py_obj.size(); i++) {
    c10::IValue payload = *py_obj_to_ivalue(py_obj[i]);
    tensors[i] = payload.toTensor();
  }
  return tensors;
}

TensorNode _get_tensor_structure(const py::sequence& py_obj) {
  // Empty list of Tensors
  if (py_obj.size() == 0) {
    return TensorNode();
  }
  if (auto tensor_sequence = to_tensor_sequence(py_obj)) {
    // List of Tensors
    return TensorNode(std::move(*tensor_sequence));
  } else {
    // List of lists of Tensors
    std::vector<TensorNode> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      py::sequence py_obj_i = py::sequence(py_obj[i]);
      result.push_back(_get_tensor_structure(py_obj_i));
    }
    return TensorNode(std::move(result));
  }
}

THPNestedTensor as_nested_tensor(py::sequence list) {
  return THPNestedTensor(_ListNestedTensor(_get_tensor_structure(list)));
}

_BufferNestedTensor make_contiguous(TensorNode _structure) {
  c10::List<at::Tensor> tensors;
  for (const at::Tensor& tensor : flatten(_structure)) {
    tensors.emplace_back(tensor.reshape({-1}));
  }
  at::Tensor buffer;
  if (tensors.size() == 0) {
    buffer = torch::ones({});
  } else {
    buffer = at::cat(tensors.vec(), 0);
  }
  SizeNode structure =
      map([](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
          _structure);
  return _BufferNestedTensor(std::move(buffer), std::move(structure));
}

} // namespace nested_tensor
} // namespace torch
