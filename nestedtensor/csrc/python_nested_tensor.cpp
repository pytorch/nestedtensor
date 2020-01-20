#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

std::vector<py::object> unbind_THPSizeNode(
    SizeNode size_node,
    std::string name) {
  std::vector<py::object> result;
  if (size_node.is_leaf()) {
    for (size_t i = 0; i < size_node.size(); i++) {
      result.push_back(torch::jit::toPyObject(size_node.payload(i)));
    }
  } else {
    for (size_t i = 0; i < size_node.degree(); i++) {
      result.push_back(py::cast(THPSizeNode(size_node.children(i), name)));
    }
  }
  return result;
}

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
