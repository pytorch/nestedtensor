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
  auto self_opt_sizes = get_opt_sizes(input);
  if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(weight)) {
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
      if (get_is_contiguous(input) && input.dtype() == torch::kHalf) {
        std::cout << "HERE" << std::endl;
        // input = input.transpose(1, 3);
        // input = NestedTensor_contiguous(input);
        Tensor input_buffer_ = get_buffer(input);
        Tensor nt_sizes_ =
            get_efficient_nested_size(input).sizes().to(torch::kInt32);
        std::cout << "nt_sizes_: " << nt_sizes_ << std::endl;
        Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1);
        Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1);
        Tensor nt_sizes_all = nt_sizes_1 * nt_sizes_2;
        int* nt_sizes_all_ptr = nt_sizes_all.data_ptr<int>();
        std::vector<int> numbers;
        numbers.reserve(1 + (nt_sizes_all.size(0) * *self_opt_sizes[1]));
        numbers.push_back(0);
        int64_t index = 1;
        for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
          for (int64_t j = 0; j < *self_opt_sizes[1]; j++) {
            numbers.push_back(numbers[index - 1] + nt_sizes_all_ptr[i]);
            index++;
          }
        }
        at::Tensor numbers_t = torch::tensor(numbers).to(torch::kInt32);
        std::cout << "numbers_t: " << numbers_t << std::endl;
        Tensor nt_sizes = numbers_t.to(torch::kCUDA);
        c10::Half* input_ptr = input_buffer_.data_ptr<c10::Half>();
        input_buffer = input_buffer_.clone();
        input_buffer.fill_(-1);
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        defaultStream.synchronize();
        c10::Half* output_ptr = input_buffer.data_ptr<c10::Half>();
        nested_tensor::cuda::transpose_kernelLauncher(
            input_ptr,
            output_ptr,
            nt_sizes.data_ptr<int>(),
            *self_opt_sizes[0],
            *self_opt_sizes[1],
            defaultStream
            );
        std::cout << "01 input_buffer_: " << input_buffer_ << std::endl;
        std::cout << "02 input_buffer: " << input_buffer << std::endl;
        input_buffer = input_buffer.reshape({-1, weight.size(1)});
      }
      {
        std::cout << "11 input_buffer: " << get_buffer(input) << std::endl;
        input = input.transpose(1, 3);
        input = NestedTensor_contiguous(input);
        input_buffer = get_buffer(input);
        std::cout << "12 input_buffer: " << input_buffer << std::endl;
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
