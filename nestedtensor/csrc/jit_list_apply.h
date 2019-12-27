#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

static bool DEBUG = false;

pybind11::cpp_function jit_tensorwise();

} // namespace nested_tensor
} // namespace torch
