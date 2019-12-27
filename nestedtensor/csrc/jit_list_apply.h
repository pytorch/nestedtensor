#include <Python.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

static bool DEBUG = false;

THPNestedTensor jit_apply_function(
    std::vector<THPNestedTensor> nts_,
    pybind11::object fn);

pybind11::cpp_function jit_tensorwise();

} // namespace nested_tensor
} // namespace torch
