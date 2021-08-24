#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <c10/util/Half.h>
#include <ATen/cuda/CUDAContext.h>
#endif
#include <nestedtensor/csrc/masking.h>
#include <nestedtensor/csrc/transpose.h>

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
  TORCH_CHECK(get_dim(input) == 4, "Expected input to be dim 4, but got ", get_dim(input), ".");
#ifdef WITH_CUDA
  auto self_opt_sizes = get_opt_sizes(input);
  if (is_nested_tensor_impl(input) &&
      !is_nested_tensor_impl(weight) &&
      (input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32)) {
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1 &&
        *self_opt_sizes[0] &&
        *self_opt_sizes[1] &&
        get_is_cuda(input)
      ) {
      if (get_is_contiguous(input, c10::MemoryFormat::ChannelsLast)) {
        Tensor input_buffer = get_buffer(input);
        input_buffer = input_buffer.view({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(input_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[0] = weight_size_0;
            }, get_efficient_nested_size(input));
        auto new_strides = map_efficient_size([] (int64_t* size_ptr, int64_t size) {
            int64_t tmp2 = size_ptr[2];
            size_ptr[2] = size_ptr[0];
            int64_t tmp1 = size_ptr[1];
            size_ptr[1] = size_ptr[2] * tmp2;
            size_ptr[0] = 1;
            }, new_sizes);
        return wrap_buffer(result_buffer.view(-1), new_sizes, new_strides);
      }
      if (get_is_contiguous(input)) {
        input = transpose_nchw_nhwc(input);
        Tensor input_buffer = get_buffer(input);
        input_buffer = input_buffer.reshape({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(input_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[2] = weight_size_0;
            }, get_efficient_nested_size(input));
        Tensor result = wrap_buffer(result_buffer.reshape(-1), new_sizes);
        return transpose_nhwc_nchw(result);
      }
    }
  }
#endif
  if (input.dtype() == torch::kFloat16) {
    at::Tensor data = to_padded_tensor(input, 0);
    at::Tensor result_data = at::conv2d(data, weight, bias, stride, padding, dilation, groups);
    auto new_sizes = map_efficient_size([&weight, &stride, &padding, &groups, &dilation](int64_t* size_ptr, int64_t size) {
        size_ptr[0] = weight.size(0);
        size_ptr[1] = ((size_ptr[1] + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) / stride[0]) + 1;
        size_ptr[2] = ((size_ptr[2] + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) / stride[1]) + 1;
        }, get_efficient_nested_size(input));
    Tensor result = from_padded_tensor(result_data, new_sizes);
    if (get_is_contiguous(input, c10::MemoryFormat::ChannelsLast)) {
      return NestedTensor_contiguous(result, c10::MemoryFormat::ChannelsLast);
    }
    return result;
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

Tensor NestedTensor_cudnn_convolution_relu(
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  Tensor input = input_;
  TORCH_CHECK(get_dim(input) == 4, "Expected input to be dim 4, but got ", get_dim(input), ".");
#ifdef WITH_CUDA
  auto self_opt_sizes = get_opt_sizes(input);
  if (is_nested_tensor_impl(input) &&
      !is_nested_tensor_impl(weight) &&
      (input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32)) {
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1 &&
        *self_opt_sizes[0] &&
        *self_opt_sizes[1] &&
        get_is_cuda(input)
      ) {
      if (get_is_contiguous(input, c10::MemoryFormat::ChannelsLast)) {
        Tensor input_buffer = get_buffer(input);
        input_buffer = input_buffer.view({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(input_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[0] = weight_size_0;
            }, get_efficient_nested_size(input));
        auto new_strides = map_efficient_size([] (int64_t* size_ptr, int64_t size) {
            int64_t tmp2 = size_ptr[2];
            size_ptr[2] = size_ptr[0];
            int64_t tmp1 = size_ptr[1];
            size_ptr[1] = size_ptr[2] * tmp2;
            size_ptr[0] = 1;
            }, new_sizes);
        return wrap_buffer(result_buffer.view(-1), new_sizes, new_strides);
      }
      if (get_is_contiguous(input)) {
        input = transpose_nchw_nhwc(input);
        Tensor input_buffer = get_buffer(input);
        input_buffer = input_buffer.reshape({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(input_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[2] = weight_size_0;
            }, get_efficient_nested_size(input));
        Tensor result = wrap_buffer(result_buffer.reshape(-1), new_sizes);
        return transpose_nhwc_nchw(result);
      }
    }
  }
#endif
  if (input.dtype() == torch::kFloat16) {
    at::Tensor data = to_padded_tensor(input, 0);
    at::Tensor result_data = at::cudnn_convolution_relu(data, weight, bias, stride, padding, dilation, groups);
    auto new_sizes = map_efficient_size([&weight, &stride, &padding, &groups, &dilation](int64_t* size_ptr, int64_t size) {
        size_ptr[0] = weight.size(0);
        size_ptr[1] = ((size_ptr[1] + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) / stride[0]) + 1;
        size_ptr[2] = ((size_ptr[2] + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) / stride[1]) + 1;
        }, get_efficient_nested_size(input));
    Tensor result = from_padded_tensor(result_data, new_sizes);
    if (get_is_contiguous(input, c10::MemoryFormat::ChannelsLast)) {
      return NestedTensor_contiguous(result, c10::MemoryFormat::ChannelsLast);
    }
    return result;
  }
  if (bias) {
      return map_nested_tensor(
          [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
            return at::cudnn_convolution_relu(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
          },
          input,
          weight,
          *bias);
  }
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight) {
      return at::cudnn_convolution_relu(input.unsqueeze(0), weight, c10::nullopt, stride, padding, dilation, groups).squeeze(0);
      },
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
  nt_impl(m, "cudnn_convolution_relu", NestedTensor_cudnn_convolution_relu);
}
} // namespace at
