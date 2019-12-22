#include <Python.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <python_list_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

// TODO Expand to IValues to support generic lists?
template <class F>
inline at::Tensor run_function(std::vector<c10::IValue> stack, F& fn);

template <>
inline at::Tensor run_function(std::vector<c10::IValue> stack, Function& fn) {
  return fn(stack).toTensor();
}

template <>
inline at::Tensor run_function(std::vector<c10::IValue> stack, Operation& fn) {
  fn(stack);
  return stack.front().toTensor();
}

THP_ListNestedTensor jit_apply_function(
    std::vector<THP_ListNestedTensor> nts_,
    py::object fn);

py::cpp_function jit_tensorwise();

} // namespace nested_tensor
} // namespace torch
