#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

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
    if (!bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1
      ) {
      input = NestedTensor_contiguous(input);
      auto input_opt_sizes = get_opt_sizes(input);
      at::Tensor input_buffer = get_buffer(input);
      at::Tensor weight_view = weight.view({weight.size(0), -1});
      std::cout << "orig input_buffer: " << input_buffer << std::endl;
      input_buffer = input_buffer.reshape({1, 1, -1, weight_view.size(1)});
      std::cout << "input_buffer.sizes(): " << input_buffer.sizes() << std::endl;
      std::cout << "input_buffer: " << input_buffer << std::endl;
      at::Tensor result = at::matmul(input_buffer, weight_view.transpose(0, 1));
      std::cout << 
        "at::matmul(input_buffer, weight_view.transpose(0, 1)): " <<
        result <<
        std::endl;
      auto ef_size = get_efficient_nested_size(input);
      auto ef_stride = get_efficient_nested_stride(input);
      auto out_size = map_efficient_size(
          [&weight](int64_t* size_ptr, int64_t size_size) {
          size_ptr[0] = weight.size(0);
          for  (int64_t i = 0; i < size_size; i++) {
          std::cout << size_ptr[i] << std::endl;
          }
          }, ef_size);
      return wrap_buffer(result.reshape(-1), 
          out_size, out_stride);

      // std::cout << 
      // "at::conv2d(input_buffer, weight, bias, stride, padding, dilation, groups): " <<
      // at::conv2d(input_buffer, weight.view(1, -1), bias, stride, padding, dilation, groups) <<
      // std::endl;
      // std::cout << 
      //   "torch::im2col(input_buffer, {weight.size(0), weight.size(1)}): " <<
      //   torch::im2col(input_buffer, {weight.size(2), weight.size(3)},
      //       {1, 1}, {0, 0}, {1, 1}) << std::endl;
      // std::cout << "input_buffer.reshape({-1, weight_view.size(0)}): " <<
      //   input_buffer.reshape({1, -1, weight_view.size(1)}).transpose(1, 2) << std::endl;
      // std::cout << "weight: " << weight << std::endl;
      // std::cout << "weight_view: " << weight_view << std::endl;
      // std::cout << "bias.has_value(): " << bias.has_value() << std::endl;
      // std::cout << "is_nested_tensor_impl(input): " << is_nested_tensor_impl(input) << std::endl;
      // std::cout << "is_nested_tensor_impl(weight): " << is_nested_tensor_impl(weight) << std::endl;
      // std::cout << "weight.sizes(): " << weight.sizes() << std::endl;
      // std::cout << "stride: " << stride << std::endl;
      // std::cout << "padding: " << padding << std::endl;
      // std::cout << "dilation: " << dilation << std::endl;
      // std::cout << "groups: " << groups << std::endl;
      // at::Tensor result = at::matmul(weight_view, input_buffer.reshape({1, -1, weight_view.size(1)}).transpose(1, 2));
      // std::cout << "result: " << result << std::endl;
    }
  }
  if (bias) {
      return map_nested_tensor(
          [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
            return at::conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
            // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
          },
          input,
          weight,
          *bias);
  }
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight) {
        return at::conv2d(input.unsqueeze(0), weight, c10::nullopt, stride, padding, dilation, groups).squeeze(0);
        // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
      },
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
