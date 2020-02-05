#include <creation.h>
#include <nested_node.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

NestedNode<c10::IValue> _get_structure(const py::object& py_obj) {
  if (py::isinstance<py::sequence>(py_obj)) {
    std::vector<NestedNode<c10::IValue>> result;
    auto py_seq = py::sequence(py_obj);
    for (size_t i = 0; i < py_seq.size(); i++) {
      result.emplace_back(_get_structure(py_seq[i]));
    }
    return NestedNode<c10::IValue>(std::move(result));
  } else {
    return NestedNode<c10::IValue>(py_obj_to_ivalue(py_obj));
  }
}

THPIValueNode as_nested_list(py::sequence list) {
  return THPIValueNode(_get_structure(list), "NestedList");
}

THPNestedTensor as_nested_tensor(py::sequence _list) {
  py::object list = _list;
  NestedNode<c10::IValue> ivalue_structure = _get_structure(list);
  auto fn = [](c10::IValue a, bool result) { return result && a.isTensor(); };
  bool all_same =
      reduce<decltype(fn), bool, c10::IValue>(ivalue_structure, fn, true);
  TORCH_CHECK(
      all_same,
      "Input nested list entries need to consist entirely of Tensors.");
  TensorNode structure =
      map([](c10::IValue a) { return a.toTensor(); }, ivalue_structure);
  if (auto first = get_first_leaf(structure)) {
    TORCH_CHECK(_verify_variables(*first, structure), "Tensors don't line up.");
  }
  return THPNestedTensor(_ListNestedTensor(std::move(structure)));
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
