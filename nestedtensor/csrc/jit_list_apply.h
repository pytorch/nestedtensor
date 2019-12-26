#include <Python.h>
#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {
THPNestedTensor jit_apply_function(
    std::vector<THPNestedTensor> nts_,
    py::object fn);
}
} // namespace torch
