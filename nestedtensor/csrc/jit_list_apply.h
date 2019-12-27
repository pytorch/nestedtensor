#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {
THPNestedTensor jit_apply_function(
    std::vector<THPNestedTensor> nts_,
    py::object fn);
}
} // namespace torch
