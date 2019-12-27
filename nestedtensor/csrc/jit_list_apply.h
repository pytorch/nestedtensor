#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

static bool DEBUG = false;

THPNestedTensor jit_apply_function(
    std::vector<THPNestedTensor> nts_,
    pybind11::object fn);

pybind11::cpp_function jit_tensorwise();

} // namespace nested_tensor
} // namespace torch
