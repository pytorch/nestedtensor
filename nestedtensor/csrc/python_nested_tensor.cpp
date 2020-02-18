#include <creation.h>
#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

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

// TODO: Since this returns vies there is no reason not to return a sequence of
// contiguous NestedTensors for a given NestedTensor.
py::object THPNestedTensor::unbind() {
  if (nested_dim() == 1) {
    return wrap_nested_node(get_structure());
  } else {
    std::vector<py::object> result;
    for (const auto& _child : get_structure().unbind()) {
      auto child = _child;
      result.push_back(py::cast(THPNestedTensor(
          torch::nested_tensor::_ListNestedTensor(std::move(child)))));
    }
    return py::cast(result);
  }
}

} // namespace nested_tensor
} // namespace torch
