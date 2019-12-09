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
  //TODO: Deal with default initialization
  _BufferNestedTensor() = delete;
  _BufferNestedTensor(torch::autograd::Variable buffer)
      : _buffer(buffer) {}
  torch::autograd::Variable get_buffer() {
    return _buffer;
  }
  int64_t element_size() {
    return _buffer.element_size();
  }

  at::ScalarType scalar_type() {
    return _buffer.scalar_type();
  }
  at::Backend backend() {
    return _buffer.type().backend();
  }
  at::Device device() {
    return _buffer.device();
  }
  at::TensorOptions options() {
    return _buffer.options();
  }
  bool requires_grad() {
    return _buffer.requires_grad();
  }

 private:
  at::Tensor _buffer;
};

} // namespace nested_tensor
} // namespace torch
