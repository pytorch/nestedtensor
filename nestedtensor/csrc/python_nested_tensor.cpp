#include <python_nested_tensor.h>

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

} // namespace nested_tensor
} // namespace torch
