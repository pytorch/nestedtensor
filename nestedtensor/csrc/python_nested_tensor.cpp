#include <creation.h>
#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

template <typename T>
THPNestedNode::THPNestedNode(SizeNode size_node, std::string name)
    : _size_node(size_node),
      _name(name),
      _elements(unbind_THPNestedNode<c10::List<int64_t>>(_size_node, _name)) {}

py::object THPNestedTensor::getDtype() {
  return data_map<py::object>(_data, [](auto data) {
    return py::reinterpret_steal<py::object>(
        torch::autograd::utils::wrap(torch::getDtype(data.scalar_type())));
  });
}

py::object THPNestedTensor::getLayout() {
  return data_map<py::object>(_data, [](auto data) {
    return py::reinterpret_steal<py::object>(
        torch::autograd::utils::wrap(torch::getLayout(data.backend())));
  });
}

py::object THPNestedTensor::getDevice() {
  return data_map<py::object>(
      _data, [](auto data) { return torch::jit::toPyObject(data.device()); });
}

THPNestedTensor THPNestedTensor::contiguous() {
  if (this->is_contiguous()) {
    return *this;
  }
  return THPNestedTensor(make_contiguous(this->get_structure()));
}

} // namespace nested_tensor
} // namespace torch
