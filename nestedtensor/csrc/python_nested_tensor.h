#pragma once

#include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/core/ivalue.h>
// #include <torch/csrc/Device.h>
// #include <torch/csrc/Dtype.h>
// #include <torch/csrc/Exceptions.h>
// #include <torch/csrc/autograd/python_variable.h>
// #include <torch/csrc/python_headers.h>
// #include <torch/csrc/tensor/python_tensor.h>
// #include <torch/csrc/utils/tensor_new.h>

// TODO:
// - HANDLE_TH_ERRORS
// - Python exception handling.
// - Implement NestedSize to avoid expensive python ops in *_nested_size/stride
// - map and apply functions that accepted JIT-ed functions to avoid unbind
// - don't make some functions static?
// - DEBUG enabled checking of constiuents

namespace torch {
namespace nested_tensor {


void initialize_python_bindings();

} // namespace nested_tensor
} // namespace torch
