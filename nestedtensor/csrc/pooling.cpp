#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_adaptive_avg_pool2d(
    at::Tensor const& input,
    IntArrayRef output_size) {
  if (is_serialized_size_node(output_size)) {
    SizeNode deserialized_size_node = deserialize_size_node(output_size);
    return wrap_tensor_node(map(
        [](const at::Tensor input, c10::List<int64_t> output_size) {
          return at::native::adaptive_avg_pool2d(
              input, IntArrayRef(output_size.vec()));
        },
        get_nested_tensor_structure(input),
        deserialized_size_node));
  }
  return wrap_tensor_node(map(
      [&output_size](at::Tensor input) {
        return at::native::adaptive_avg_pool2d(input, output_size);
      },
      get_nested_tensor_structure(input)));
}

Tensor NestedTensor_adaptive_avg_pool2d_backward(
    const Tensor& gradInput,
    const Tensor& input) {
  return map_nested_tensor(
      [](at::Tensor gradInput, at::Tensor input) {
        return at::_adaptive_avg_pool2d_backward(gradInput, input);
      },
      gradInput,
      input);
}

Tensor NestedTensor_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  return map_nested_tensor(
      [&](at::Tensor t) {
        return at::max_pool2d(
                   t.unsqueeze(0),
                   kernel_size,
                   stride,
                   padding,
                   dilation,
                   ceil_mode)
            .squeeze(0);
      },
      self);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "max_pool2d", NestedTensor_max_pool2d);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "adaptive_avg_pool2d", NestedTensor_adaptive_avg_pool2d);
  nt_impl(
      m,
      "adaptive_avg_pool2d_backward",
      NestedTensor_adaptive_avg_pool2d_backward);
}

} // namespace at
