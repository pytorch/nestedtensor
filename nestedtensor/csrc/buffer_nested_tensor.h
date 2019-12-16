#pragma once
#include <Python.h>
#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

// Make _structure initialization lazy?
struct TORCH_API _BufferNestedTensor {
  // TODO: Deal with default initialization
  _BufferNestedTensor() = delete;
  _BufferNestedTensor(torch::autograd::Variable buffer, SizeNode nested_size);
  _BufferNestedTensor(
      torch::autograd::Variable buffer,
      SizeNode nested_size,
      SizeNode nested_stride);
  _BufferNestedTensor(
      torch::autograd::Variable buffer,
      SizeNode nested_size,
      SizeNode nested_stride,
      TensorNode structure);
  torch::autograd::Variable get_buffer() {
    return _buffer;
  }
  int64_t element_size() {
    return _buffer.element_size();
  }
  int64_t __len__() {
    if (nested_dim() == 1) {
      return _nested_size.size();
    } else {
      return _nested_size.degree();
    }
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
  _BufferNestedTensor requires_grad_(bool requires_grad) {
    apply<at::Tensor>(_structure, [requires_grad](at::Tensor tensor) {
      tensor.set_requires_grad(requires_grad);
    });
    _buffer.set_requires_grad(requires_grad);
    return *this;
  }
  bool requires_grad() {
    return _buffer.requires_grad();
  }
  _BufferNestedTensor grad() {
    at::Tensor grad_buffer = _buffer.grad();
    TensorNode grad_tensors = map<at::Tensor, at::Tensor>(
        _structure,
        [](at::Tensor tensor) -> at::Tensor { return tensor.grad(); });
    return _BufferNestedTensor(
        grad_buffer, _nested_size, _nested_stride, grad_tensors);
  }
  // TODO: This should return a reference?
  _BufferNestedTensor detach() {
    at::Tensor detach_buffer = _buffer.detach();
    TensorNode detach_tensors = map<at::Tensor, at::Tensor>(
        _structure,
        [](at::Tensor tensor) -> at::Tensor { return tensor.detach(); });
    return _BufferNestedTensor(
        detach_buffer, _nested_size, _nested_stride, detach_tensors);
  }
  bool is_pinned() {
    return _buffer.is_pinned();
  }
  SizeNode nested_size() {
    return _nested_size;
  }
  SizeNode nested_stride() {
    return _nested_stride;
  }
  TensorNode get_structure() {
    return _structure;
  }
  int64_t nested_dim() {
    const SizeNode* start_structure = &_nested_size;
    int64_t depth = 1;
    while (!start_structure->is_leaf()) {
      depth++;
      start_structure = start_structure->children_data(0);
    }
    return depth;
  }
  int64_t dim() {
    SizeNode leaf = get_first_leaf(_nested_size);
    if (leaf.degree()) {
      c10::List<int64_t> first_size = leaf.payload(0);
      return first_size.size() + nested_dim();
    } else {
      return nested_dim();
    }
  }

 private:
  at::Tensor _buffer;
  SizeNode _nested_size;
  SizeNode _nested_stride;
  TensorNode _structure;
};

} // namespace nested_tensor
} // namespace torch
