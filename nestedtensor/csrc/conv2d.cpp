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
  if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(weight) && input.dtype() == torch::kFloat16) {
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1 &&
        *self_opt_sizes[0] &&
        *self_opt_sizes[1] &&
        get_is_cuda(input)
      ) {
      at::Tensor input_buffer;
      int64_t weight_size_0 = weight.size(0);
      auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
          std::cout << "size: " << size << std::endl;
            std::cout << ", 0 conv2d : ";
            for (int64_t i = 0; i < size; i++) {
            std::cout << ", " << size_ptr[i];
            }
          size_ptr[0] = weight_size_0;
            std::cout << ", 1 conv2d : ";
            for (int64_t i = 0; i < size; i++) {
            std::cout << ", " << size_ptr[i];
            }
          }, get_efficient_nested_size(input));
        std::cout << std::endl;
      if (get_is_channel_last(input) && input.dtype() == torch::kHalf) {
        Tensor input_buffer = get_buffer_channel_last(input);
        input_buffer = input_buffer.reshape({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(input_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        return wrap_buffer_channel_last(result_buffer.reshape(-1), new_sizes);
      }
      if (get_is_contiguous(input) && input.dtype() == torch::kHalf) {
        // Tensor nt_sizes =
        //     get_efficient_nested_size(input).sizes();
        // Tensor nt_sizes_0 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
        // Tensor nt_sizes_1 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
        // Tensor nt_sizes_2 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();
        // Tensor nt_sizes_1_2 = nt_sizes_1 * nt_sizes_2;
        // nt_sizes = at::cat({nt_sizes_0, nt_sizes_1_2}, 1);
        // Tensor input_buffer = get_buffer(input);
        // Tensor output_buffer = input_buffer.clone();
        // output_buffer = transpose_buffer(nt_sizes, input_buffer, output_buffer);
        Tensor output_buffer = get_buffer_channel_last(transpose_nchw_nhwc(input));
        output_buffer = output_buffer.reshape({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(output_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        // nt_sizes_0.fill_(weight_size_0);
        // nt_sizes = at::cat({nt_sizes_1_2, nt_sizes_0}, 1);
        // output_buffer.resize_as_(result_buffer);
        // output_buffer = transpose_buffer(nt_sizes,
        //                                  result_buffer.reshape(-1),
        //                                  output_buffer.reshape(-1));
        // return wrap_buffer(output_buffer.reshape(-1), new_sizes);
        return wrap_buffer_channel_last(result_buffer.reshape(-1), new_sizes);
      }
    }
  }
#endif
  if (input.dtype() == torch::kFloat16) {
    // std::cout << "start conv2d" << std::endl;
    // bool got_channel_last = false;
    // if (get_is_channel_last(input)) {
    //     got_channel_last = true;
    //     at::Tensor data = to_padded_tensor(input, 0);
    //     std::cout << "data.sizes(): " << data.sizes() << std::endl;
    //     std::cout << "data.strides(): " << data.strides() << std::endl;
    //     // input = transpose_nhwc_nchw(input);
    // }
    at::Tensor data = to_padded_tensor(input, 0);
    std::cout << "conved 0 data.sizes(): " << data.sizes() << std::endl;
    std::cout << "conved 0 data.strides(): " << data.strides() << std::endl;
    // if (get_is_channel_last(input)) {
    //   data = data.permute({0, 3, 1, 2});
    // }
    std::cout << "conved 1 data.sizes(): " << data.sizes() << std::endl;
    std::cout << "conved 1 data.strides(): " << data.strides() << std::endl;
    at::Tensor result_data = at::conv2d(data, weight, bias, stride, padding, dilation, groups);
    auto new_sizes = map_efficient_size([&weight, &stride, &padding, &groups, &dilation](int64_t* size_ptr, int64_t size) {
            std::cout << "- 0 new conv2d : ";
            for (int64_t i = 0; i < size; i++) {
            std::cout << ", " << size_ptr[i];
            }
        size_ptr[0] = weight.size(0);
        size_ptr[1] = ((size_ptr[1] + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) / stride[0]) + 1;
        size_ptr[2] = ((size_ptr[2] + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) / stride[1]) + 1;
            std::cout << "- 1 new conv2d : ";
            for (int64_t i = 0; i < size; i++) {
            std::cout << ", " << size_ptr[i];
            }
        }, get_efficient_nested_size(input));
    std::cout << std::endl;
    // if (!get_is_contiguous(result_data)) {
    //   std::cout << "result_data.sizes(): " << result_data.sizes() << std::endl;
    //   std::cout << "result_data.strides(): " << result_data.strides() << std::endl;
    //   std::cout << "HERE" << std::endl;
    //   exit(1);
    // }
    // if (got_channel_last) {
    //   std::cout << "result_data.sizes(): " << result_data.sizes() << std::endl;
    //   std::cout << "result_data.strides(): " << result_data.strides() << std::endl;
    // }
    at::Tensor result = from_padded_tensor(result_data, new_sizes);
    // if (got_channel_last) {
    //   std::cout << "result.sizes(): " << result.sizes() << std::endl;
    //   std::cout << "result.strides(): " << result.strides() << std::endl;
    //   exit(1);
    //   return transpose_nchw_nhwc(result);
    // }
    // std::cout << "end conv2d" << std::endl;
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

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
