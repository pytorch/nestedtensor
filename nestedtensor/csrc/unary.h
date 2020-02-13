#include <creation.h>
#include <python_nested_tensor.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

void add_unary_functions(
    pybind11::module,
    pybind11::class_<torch::nested_tensor::THPNestedTensor>);

} // namespace nested_tensor
} // namespace torch
