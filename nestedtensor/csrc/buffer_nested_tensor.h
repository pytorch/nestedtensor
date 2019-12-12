#pragma once
#include <Python.h>
#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

_NestedNode _infer_stride(_NestedNode nested_size) {
  if (nested_size.is_leaf()) {
    auto stride = nested_size.payload().toIntList().copy();
    return _NestedNode(stride);
  } else {
    std::vector<_NestedNode> result;
    for (size_t i = 0; i < nested_size.degree(); i++) {
      result.push_back(_infer_stride(nested_size.children(i)));
    }
    return _NestedNode(result);
  }
}


// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct TORCH_API _BufferNestedTensor {
  // TODO: Deal with default initialization
  _BufferNestedTensor() = delete;
  _BufferNestedTensor(
      torch::autograd::Variable buffer,
      _NestedNode nested_size)
      : _buffer(buffer),
        _nested_size(nested_size),
        _nested_stride(_infer_stride(nested_size)) {}
  _BufferNestedTensor(
      torch::autograd::Variable buffer,
      _NestedNode nested_size,
      _NestedNode nested_stride)
      : _buffer(buffer),
        _nested_size(nested_size),
        _nested_stride(nested_stride) {}
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
  _NestedNode nested_size() {
    return _nested_size;
  }
  _NestedNode nested_stride() {
    return _nested_stride;
  }

 private:
  at::Tensor _buffer;
  _NestedNode _nested_size;
  _NestedNode _nested_stride;
};

} // namespace nested_tensor
} // namespace torch
