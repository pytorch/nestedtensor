#pragma once
#include <nested_node.h>

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
  _BufferNestedTensor(
      torch::autograd::Variable&& buffer,
      SizeNode nested_size,
      SizeNode nested_stride,
      TensorNode&& structure);
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
  _BufferNestedTensor grad();
  // TODO: This should return a reference?
  _BufferNestedTensor detach() {
    at::Tensor detach_buffer = _buffer.detach();
    TensorNode detach_tensors = map<at::Tensor, at::Tensor>(
        _structure,
        [](at::Tensor tensor) -> at::Tensor { return tensor.detach(); });
    return _BufferNestedTensor(
        detach_buffer, _nested_size, _nested_stride, detach_tensors);
  }
  _BufferNestedTensor pin_memory();
  void backward(
      _BufferNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    // TODO: This should be enough due to split.
    _buffer.backward(gradient.get_buffer(), retain_graph, create_graph);
  }
  bool is_pinned() {
    return _buffer.is_pinned();
  }
  bool is_contiguous() {
    // NOTE: The Tensors might not be contiguous themselves.
    // For this to be contiguous not only do the Tensors need to
    // come from the buffer, but they also need to
    return all_contiguous(_structure);
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
    const TensorNode* start_structure = &_structure;
    int64_t depth = 1;
    while (!start_structure->is_leaf()) {
      depth++;
      start_structure = start_structure->children_data(0);
    }
    return depth;
  }
  int64_t dim() {
    if (const auto& maybe_tensor = get_first_leaf(_structure)) {
      return (*maybe_tensor).dim() + nested_dim();
    }
    return nested_dim();
  }
  int64_t numel() {
    return nested_node_numel(_structure);
  }
  at::Tensor to_tensor() {
    auto size = construct_size(_nested_size);
    std::vector<int64_t> new_size(size.size());
    for (size_t i = 0; i < size.size(); i++) {
      if (!size[i]) {
        throw std::runtime_error("to_tensor only works if all sizes equal.");
      }
      new_size[i] = *size[i];
    }
    return _buffer.reshape(at::IntArrayRef(new_size));
  }

 private:
  at::Tensor _buffer;
  const SizeNode _nested_size;
  const SizeNode _nested_stride;
  const TensorNode _structure;
};

} // namespace nested_tensor
} // namespace torch
