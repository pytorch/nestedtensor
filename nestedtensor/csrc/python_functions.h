#pragma once	
// #include <nestedtensor/csrc/python_nested_tensor.h>

namespace torch {
namespace nested_tensor {
void add_functions(pybind11::module);
} // namespace nested_tensor
} // namespace torch
