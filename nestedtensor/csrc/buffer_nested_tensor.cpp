#include <buffer_nested_tensor.h>

namespace torch {
namespace nested_tensor {

std::vector<int64_t> _cont_stride(c10::List<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return stride;
}

SizeNode _infer_stride(SizeNode nested_size) {
  if (nested_size.is_leaf()) {
    c10::List<c10::List<int64_t>> result;
    for (size_t i = 0; i < nested_size.size(); i++) {
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
      auto size_i = nested_size.payload(i).vec();
      auto stride_i = nested_stride.payload(i).vec();
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
      index = std::get<0>(result_i);
      result.push_back(std::get<1>(result_i));
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

// TODO: Test this. Does split on pinned memory work?
_BufferNestedTensor _BufferNestedTensor::pin_memory() {
  at::Tensor new_buffer = _buffer.pin_memory();
  TensorNode new_tensor_node =
      build_structure(new_buffer, _nested_size, _nested_stride);
  return _BufferNestedTensor(
      new_buffer, _nested_size, _nested_stride, new_tensor_node);
}

_BufferNestedTensor _BufferNestedTensor::grad() {
  at::Tensor grad_buffer = _buffer.grad();
  // TODO: TensorNodes are based on split. Any backward performed on those will
  // accumulate in the buffer's grad. What we're creating here are views into
  // the grad, which could then be used further.
  TensorNode grad_tensor_node =
      build_structure(grad_buffer, _nested_size, _nested_stride);
  return _BufferNestedTensor(
      grad_buffer, _nested_size, _nested_stride, grad_tensor_node);
}

_BufferNestedTensor::_BufferNestedTensor(
    torch::autograd::Variable buffer,
    SizeNode nested_size)
    : _BufferNestedTensor(buffer, nested_size, _infer_stride(nested_size)) {}
_BufferNestedTensor::_BufferNestedTensor(
    torch::autograd::Variable buffer,
    SizeNode nested_size,
    SizeNode nested_stride)
    : _BufferNestedTensor(
          buffer,
          nested_size,
          nested_stride,
          build_structure(buffer, nested_size, nested_stride)) {}
_BufferNestedTensor::_BufferNestedTensor(
    torch::autograd::Variable buffer,
    SizeNode nested_size,
    SizeNode nested_stride,
    TensorNode structure)
    : _buffer(buffer),
      _nested_size(nested_size),
      _nested_stride(nested_stride),
      _structure(structure) {}

} // namespace nested_tensor
} // namespace torch
