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
THP_ListNestedTensor jit_apply_function(
    std::vector<THP_ListNestedTensor> nts_,
    py::object fn);
py::cpp_function jit_tensorwise();
at::Tensor resolve_builtin(
    py::object obj,
    py::args py_args,
    py::kwargs py_kwargs = {});
} // namespace nested_tensor
} // namespace torch
