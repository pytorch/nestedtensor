#pragma once
#include <Python.h>
#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

std::vector<int64_t> _cont_stride(c10::IntArrayRef size) {
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
    c10::List<c10::IntArrayRef> result;
    for (size_t i = 0; i < nested_size.degree(); i++) {
      std::vector<int64_t> stride = _cont_stride(nested_size.payload(i);
      c10::IntArrayRef stride(stride.data<int64_t>(), stride.size());
      result.push_back(stride);
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

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct TORCH_API _BufferNestedTensor {
  // TODO: Deal with default initialization
  _BufferNestedTensor() = delete;
  _BufferNestedTensor(torch::autograd::Variable buffer, SizeNode nested_size)
      : _buffer(buffer),
        _nested_size(nested_size),
        _nested_stride(_infer_stride(nested_size)) {}
  _BufferNestedTensor(
      torch::autograd::Variable buffer,
      SizeNode nested_size,
      SizeNode nested_stride)
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
  bool is_pinned() {
    return _buffer.is_pinned();
  }
  SizeNode nested_size() {
    return _nested_size;
  }
  SizeNode nested_stride() {
    return _nested_stride;
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
      c10::IntArrayRef first_size = leaf.payload(0);
      return first_size.size() + nested_dim();
    } else {
      return nested_dim();
    }
  }

 private:
  at::Tensor _buffer;
  SizeNode _nested_size;
  SizeNode _nested_stride;
};

} // namespace nested_tensor
} // namespace torch
