#include <buffer_nested_tensor.h>

namespace torch {
namespace nested_tensor {


// // TODO: Test this. Does split on pinned memory work?
// _BufferNestedTensor _BufferNestedTensor::pin_memory() {
//   at::Tensor new_buffer = _buffer.pin_memory();
//   SizeNode nested_size = _nested_size;
//   SizeNode nested_stride = _nested_stride;
//   TensorNode new_tensor_node =
//       build_structure(new_buffer, _nested_size, _nested_stride);
//   return _BufferNestedTensor(
//       std::move(new_buffer),
//       std::move(nested_size),
//       std::move(nested_stride),
//       std::move(new_tensor_node));
// }
// 
// _BufferNestedTensor _BufferNestedTensor::grad() {
//   at::Tensor grad_buffer = _buffer.grad();
//   SizeNode nested_size = _nested_size;
//   SizeNode nested_stride = _nested_stride;
//   // TODO: TensorNodes are based on split. Any backward performed on those will
//   // accumulate in the buffer's grad. What we're creating here are views into
//   // the grad, which could then be used further.
//   TensorNode grad_tensor_node =
//       build_structure(grad_buffer, _nested_size, _nested_stride);
//   return _BufferNestedTensor(
//       std::move(grad_buffer),
//       std::move(nested_size),
//       std::move(nested_stride),
//       std::move(grad_tensor_node));
// }
// 
// _BufferNestedTensor::_BufferNestedTensor(
//     torch::autograd::Variable buffer,
//     SizeNode nested_size)
//     : _BufferNestedTensor(
//           buffer,
//           nested_size,
//           map([](c10::List<int64_t> size) { return _cont_stride(size); },
//               nested_size)) {}
// _BufferNestedTensor::_BufferNestedTensor(
//     torch::autograd::Variable buffer,
//     SizeNode nested_size,
//     SizeNode nested_stride)
//     : _BufferNestedTensor(
//           buffer,
//           nested_size,
//           nested_stride,
//           build_structure(buffer, nested_size, nested_stride)) {}
// _BufferNestedTensor::_BufferNestedTensor(
//     torch::autograd::Variable buffer,
//     SizeNode nested_size,
//     SizeNode nested_stride,
//     TensorNode structure)
//     : _buffer(buffer),
//       _nested_size(nested_size),
//       _nested_stride(nested_stride),
//       _structure(structure) {}

} // namespace nested_tensor
} // namespace torch
