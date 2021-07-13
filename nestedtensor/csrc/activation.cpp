#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_gelu(const Tensor& self) {
  if (is_nested_tensor_impl(self) && get_is_contiguous(self)) {
    return wrap_buffer(
        at::gelu(get_buffer(self)),
        get_efficient_nested_size(self),
        get_efficient_nested_stride(self));
  }
  return map_nested_tensor(
      [](at::Tensor tensor) { return at::gelu(tensor); }, self);
}

Tensor NestedTensor_elu(const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  if (is_nested_tensor_impl(self) && get_is_contiguous(self)) {
    return wrap_buffer(
        at::elu(get_buffer(self), alpha, scale, input_scale),
        get_efficient_nested_size(self),
        get_efficient_nested_stride(self));
  }
  return map_nested_tensor(
      [&alpha, &scale, &input_scale](at::Tensor tensor) { return at::elu(tensor, alpha, scale, input_scale); }, self);
}

// Registered below autograd
Tensor NestedTensor_relu(const Tensor& self) {
  auto impl = get_nested_tensor_impl(self);
  auto structure = get_nested_tensor_structure(self);
  if (get_is_contiguous(self)) {
#ifdef TRACEPACKED
    std::cout << "calling packed relu" << std::endl;
#endif
    return wrap_buffer(at::relu(get_buffer(self)),
        get_efficient_nested_size(self),
        get_efficient_nested_stride(self));
  }
  return map_nested_tensor(
      [](at::Tensor tensor) { return at::relu(tensor); }, self);
}

// Registered below autograd
Tensor& NestedTensor_relu_(Tensor& self) {
  if (get_is_contiguous(self) || get_is_contiguous(self, c10::MemoryFormat::ChannelsLast)) {
#ifdef TRACEPACKED
    std::cout << "calling packed relu_" << std::endl;
#endif
    Tensor buffer = get_buffer(self);
    at::relu_(buffer);
    return self;
  }
  apply_nested_tensor([](at::Tensor& tensor) { at::relu_(tensor); }, self);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "gelu", NestedTensor_gelu);
  nt_impl(m, "elu", NestedTensor_elu);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "relu", NestedTensor_relu);
  nt_impl(m, "relu_", NestedTensor_relu_);
}

} // namespace at
