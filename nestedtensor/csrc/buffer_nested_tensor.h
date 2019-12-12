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
    c10::List<int64_t> size = nested_size.payload().toIntList();
    std::vector<int64_t> stride_;
    int64_t p = 1;
    for (int64_t i = 0; i < size.size(); i++) {
      stride_.push_back(p);
      p *= size[i];
    }
    c10::List<int64_t> stride(stride_);
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
  bool is_pinned() {
    return _buffer.is_pinned();
  }
  _NestedNode nested_size() {
    return _nested_size;
  }
  _NestedNode nested_stride() {
    return _nested_stride;
  }
  int64_t nested_dim() {
    const _NestedNode* start_structure = &_nested_size;
    int64_t depth = 1;
    while (!start_structure->is_leaf()) {
      depth++;
      start_structure = start_structure->children_data(0);
    }
    return depth;
  }
  int64_t dim() {
    const _NestedNode* start_structure = &_nested_size;
    while (!start_structure->is_leaf()) {
      start_structure = start_structure->children_data(0);
    }
    int64_t tensor_dim = start_structure->payload().toIntList().size();
    return tensor_dim + nested_dim();
  }

 private:
  at::Tensor _buffer;
  _NestedNode _nested_size;
  _NestedNode _nested_stride;
};

} // namespace nested_tensor
} // namespace torch
