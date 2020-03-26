#pragma once
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

void add_functions(
    pybind11::module,
    pybind11::class_<torch::nested_tensor::THPNestedTensor>);

}
} // namespace torch
