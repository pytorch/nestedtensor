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

void _split_sizes(
    SizeNode nested_size,
    SizeNode nested_stride,
    std::vector<int64_t>& sizes) {
  if (nested_stride.is_leaf()) {
    for (size_t i = 0; i < nested_stride.size(); i++) {
      sizes.push_back(
          num_memory(nested_size.payload(i), nested_stride.payload(i)));
    }
  } else {
    for (size_t i = 0; i < nested_size.degree(); i++) {
      _split_sizes(nested_size.children(i), nested_stride.children(i), sizes);
    }
  }
}

std::pair<int64_t, TensorNode> _build_structure(
    int64_t index,
    std::vector<at::Tensor>& buffers,
    SizeNode nested_size,
    SizeNode nested_stride) {
  if (nested_size.is_leaf()) {
    c10::List<at::Tensor> result;
    for (size_t i = 0; i < nested_size.size(); i++) {
      auto size_i = c10::impl::toVector(nested_size.payload(i));
      auto stride_i = c10::impl::toVector(nested_stride.payload(i));
      result.push_back(at::as_strided(
          buffers[index],
          c10::IntArrayRef(size_i),
          c10::IntArrayRef(stride_i)));
      index++;
    }
    return std::pair<int64_t, TensorNode>(index, TensorNode(result));
  } else {
    std::vector<TensorNode> result;
    for (size_t i = 0; i < nested_size.degree(); i++) {
      std::pair<int64_t, TensorNode> result_i = _build_structure(
          index, buffers, nested_size.children(i), nested_stride.children(i));
      result.push_back(std::get<1>(result_i));
      index++;
    }
    return std::pair<int64_t, TensorNode>(index, TensorNode(result));
  }
}

TensorNode build_structure(
    at::Tensor buffer,
    SizeNode nested_size,
    SizeNode nested_stride) {
  std::vector<int64_t> split_sizes;
  _split_sizes(nested_size, nested_stride, split_sizes);
  std::vector<int64_t> nonzero_split_sizes;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      nonzero_split_sizes.push_back(split_sizes[i]);
    }
  }
  std::vector<at::Tensor> buffers_;
  if (nonzero_split_sizes.size() > 0) {
    buffers_ =
        at::split_with_sizes(buffer, c10::IntArrayRef(nonzero_split_sizes), 0);
  }
  std::vector<at::Tensor> buffers;
  int64_t index = 0;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      buffers.push_back(buffers_[index]);
      index++;
    } else {
      buffers.push_back(at::empty({}, buffer.options()));
    }
  }
  std::pair<int64_t, TensorNode> result =
      _build_structure(0, buffers, nested_size, nested_stride);
  return std::get<1>(result);
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
        _structure(build_structure(_buffer, _nested_size, _nested_stride)) {}
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
