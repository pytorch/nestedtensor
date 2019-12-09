#include <buffer_nested_tensor.h>
// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;

struct THP_BufferNestedTensor {
  THP_BufferNestedTensor() = delete;
  THP_BufferNestedTensor(py::object buffer)
      : _data(_BufferNestedTensor(py::cast<at::Tensor>(buffer))) {}
  torch::autograd::Variable get_buffer() {
    return _data.get_buffer();
  }
  int64_t element_size() {
    return _data.element_size();
  }

  py::object getDtype() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getDtype(_data.scalar_type())));
  }
  py::object getLayout() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getLayout(_data.backend())));
  }
  py::object getDevice() {
    return toPyObject(_data.device());
  }
  bool requires_grad() {
    return _data.requires_grad();
  }


 private:
  _BufferNestedTensor _data;
};

} // namespace nested_tensor
} // namespace torch
