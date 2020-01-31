#include <buffer_nested_tensor.h>

namespace torch {
namespace nested_tensor {

c10::List<int64_t> _cont_stride(c10::List<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return c10::List<int64_t>(stride);
}

TensorNode build_structure(
    at::Tensor buffer,
    SizeNode nested_size,
    SizeNode nested_stride) {
  c10::List<int64_t> split_sizes = flatten(
      map([](c10::List<int64_t> a,
             c10::List<int64_t> b) { return num_memory(a, b); },
          nested_size,
          nested_stride));
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
  TensorNode tmp = unflatten(nested_size, c10::List<at::Tensor>(buffers));
  return map(
      [](at::Tensor buffer,
         c10::List<int64_t> size,
         c10::List<int64_t> stride) {
        return at::as_strided(
            buffer,
            c10::IntArrayRef(size.vec()),
            c10::IntArrayRef(stride.vec()));
      },
      tmp,
      nested_size,
      nested_stride);
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
    : _BufferNestedTensor(
          buffer,
          nested_size,
          map([](c10::List<int64_t> size) { return _cont_stride(size); },
              nested_size)) {}
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
