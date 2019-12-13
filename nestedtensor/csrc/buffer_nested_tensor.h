#pragma once
#include <Python.h>
#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

std::vector<int64_t> _cont_stride(c10::List<int64_t> size) {
  std::vector<int64_t> stride;
  int64_t p = 1;
  for (size_t i = 0; i < size.size(); i++) {
    stride.push_back(p);
    p *= size[i];
  }
  return stride;
}

SizeNode _infer_stride(SizeNode nested_size) {
  if (nested_size.is_leaf()) {
    c10::List<c10::List<int64_t>> result;
    for (size_t i = 0; i < nested_size.degree(); i++) {
      // std::vector<int64_t> stride = _cont_stride(nested_size.payload(i);
      // c10::IntArrayRef stride(stride.data<int64_t>(), stride.size());
      result.emplace_back(_cont_stride(nested_size.payload(i)));
    }
    return SizeNode(result);
  } else {
    std::vector<SizeNode> result;
    for (size_t i = 0; i < nested_size.degree(); i++) {
      result.push_back(_infer_stride(nested_size.children(i)));
    }
    return SizeNode(result);
  }
}

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  if (size.size() == 0) {
    return 0;
  }
  return size[0] * stride[0];
}

TensorNode _build_structure(
    at::Tensor buffer,
    SizeNode nested_size,
    SizeNode nested_stride) {
  if (nested_stride.is_leaf()) {
    std::vector<int64_t> split_sizes;
    for (size_t i = 0; i < nested_stride.size()) {
      split_sizes.push_back(nested_size.payload(i), nested_stride.payload(i));
    }
    std::vector<at::Tensor> buffers =
        at::split_with_sizes(buffer, split_sizes, 0);
    std::vector<at::Tensor> result;
    for (size_t i = 0; i < buffers.size(); i++) {
      at::as_strided(buffers[i], nested_size.payload(i), nested_stride.payload(i));
    }
    return TensorNode(result);
  } else {
    std::vector<TensorNode> result;
    for (size_t i = 0; i < nested_size.degree(); i++) {
      result.push_back(_build_structure(nested_size.children(i)));
    }
    return TensorNode(result);
  }
}

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct TORCH_API _BufferNestedTensor {
  // TODO: Deal with default initialization
  _BufferNestedTensor() = delete;
  _BufferNestedTensor(torch::autograd::Variable buffer, SizeNode nested_size)
      : _BufferNestedTensor(buffer, nested_size, _infer_stride(nested_size)) {}
  _BufferNestedTensor(
      torch::autograd::Variable buffer,
      SizeNode nested_size,
      SizeNode nested_stride)
      : _buffer(buffer),
        _nested_size(nested_size),
        _nested_stride(nested_stride),
        _structure(_build_structure(_buffer, _nested_size, _nested_stride)) {}
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
