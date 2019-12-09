#pragma once
#include <nested_node.h>
#include <Python.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct TORCH_API _BufferNestedTensor {
  _BufferNestedTensor() = delete;
  _BufferNestedTensor(torch::autograd::Variable buffer)
      : _buffer(buffer) {}
 private:
  at::Tensor _buffer;
};

} // namespace nested_tensor
} // namespace torch
