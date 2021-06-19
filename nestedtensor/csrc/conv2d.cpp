#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <c10/util/Half.h>
#endif

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_conv2d(
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  Tensor input = input_;
  if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(weight)) {
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1
      ) {
      at::Tensor input_buffer;
      std::cout << "get_is_contiguous(input): " << get_is_contiguous(input) << std::endl;
      if (get_is_contiguous(input) && input.dtype() == torch::kHalf) {
        std::cout << "HERE" << std::endl;
        input = input.transpose(1, 3);
        input = NestedTensor_contiguous(input);
        input_buffer = get_buffer(input);
        input_buffer = input_buffer.reshape({-1, weight.size(1)});
      } else {
        input = input.transpose(1, 3);
        input = NestedTensor_contiguous(input);
        input_buffer = get_buffer(input);
        input_buffer = input_buffer.reshape({-1, weight.size(1)});
      }
      at::Tensor result_buffer = at::matmul(input_buffer, 
          weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
      int64_t weight_size_0 = weight.size(0);
      auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
          size_ptr[2] = weight_size_0;
          }, get_efficient_nested_size(input));
      at::Tensor result = wrap_buffer(result_buffer.reshape(-1),
          new_sizes);
      result = result.transpose(1, 3);
      result = NestedTensor_contiguous(result);
      return result;
    }
  }
  if (bias) {
      return map_nested_tensor(
          [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
            return at::conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
          },
          input,
          weight,
          *bias);
  }
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight) {
        return at::conv2d(input.unsqueeze(0), weight, c10::nullopt, stride, padding, dilation, groups).squeeze(0);
      },
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
