#include <Python.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>
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
  THP_BufferNestedTensor(torch::autograd::Variable buffer) : _buffer(buffer) {}

 private:
  _buffer torch::autograd::Variable;
};


} // namespace nested_tensor
} // namespace torch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  py::class_<torch::nested_tensor::THP_BufferNestedTensor>(m, "_BufferNestedTensor")
      .def(py::init<py::object>(), py::keep_alive<1, 2>());
}
