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
        Tensor nt_sizes_ =
            get_efficient_nested_size(input).sizes().to(torch::kInt32);
        Tensor nt_sizes_0 = at::native::narrow(nt_sizes_, 1, 0, 1).contiguous();
        Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1).contiguous();
        Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1).contiguous();
        Tensor nt_sizes_1_2 = nt_sizes_1 * nt_sizes_2;
        Tensor nt_sizes_all = nt_sizes_0 * nt_sizes_1_2;
        int* nt_sizes_all_ptr = nt_sizes_all.data_ptr<int>();
        std::vector<int> numbers;
        numbers.reserve(1 + nt_sizes_all.size(0));
        numbers.push_back(0);
        int64_t index = 1;
        for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
          numbers.push_back(numbers[index - 1] + nt_sizes_all_ptr[i]);
          index++;
        }
        at::Tensor numbers_t = torch::tensor(numbers).to(torch::kInt32);
        // std::cout << "numbers_t: " << numbers_t << std::endl;
        Tensor nt_sizes = numbers_t.to(at::Device(kCUDA), torch::kInt32, true, true);

        nt_sizes_1_2 = nt_sizes_1_2.to(at::Device(kCUDA), torch::kInt32, true, true);
        nt_sizes_0 = (nt_sizes_0).to(at::Device(kCUDA), torch::kInt32, true, true);

        Tensor input_buffer = get_buffer(input);
        Tensor output_buffer = input_buffer.clone();
        c10::Half* input_ptr = input_buffer.data_ptr<c10::Half>();
        c10::Half* output_ptr = output_buffer.data_ptr<c10::Half>();
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        nested_tensor::cuda::transpose_kernelLauncher(
            input_ptr,
            output_ptr,
            nt_sizes.data_ptr<int>(),
            nt_sizes_0.data_ptr<int>(),
            nt_sizes_1_2.data_ptr<int>(),
            *self_opt_sizes[0],
            defaultStream
            );
        output_buffer = output_buffer.reshape({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(output_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        c10::Half* result_ptr = result_buffer.data_ptr<c10::Half>();
        int64_t weight_size_0 = weight.size(0);

        nt_sizes_0 = at::native::narrow(nt_sizes_, 1, 0, 1).contiguous();
        nt_sizes_0.fill_(weight_size_0);
        nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1).contiguous();
        nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1).contiguous();
        nt_sizes_all = nt_sizes_0 * nt_sizes_1 * nt_sizes_2;
        nt_sizes_all_ptr = nt_sizes_all.data_ptr<int>();
        numbers.clear();
        numbers.reserve(1 + nt_sizes_all.size(0));
        numbers.push_back(0);
        index = 1;
        for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
          numbers.push_back(numbers[index - 1] + nt_sizes_all_ptr[i]);
          index++;
        }
        numbers_t = torch::tensor(numbers).to(torch::kInt32);
        // std::cout << "numbers_t: " << numbers_t << std::endl;
        nt_sizes = numbers_t.to(at::Device(kCUDA), torch::kInt32, true, true);
        // nt_sizes_2 = (nt_sizes_2 * nt_sizes_1).to(at::Device(kCUDA), torch::kInt32, true, true);
        nt_sizes_0 = (nt_sizes_0).to(at::Device(kCUDA), torch::kInt32, true, true);
        result_buffer = result_buffer.reshape(-1);
        output_buffer.resize_as_(result_buffer);
        output_ptr = output_buffer.data_ptr<c10::Half>();

        nested_tensor::cuda::transpose_kernelLauncher(
            result_ptr,
            output_ptr,
            nt_sizes.data_ptr<int>(),
            nt_sizes_1_2.data_ptr<int>(),
            nt_sizes_0.data_ptr<int>(),
            *self_opt_sizes[0],
            defaultStream
            );
        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[0] = weight_size_0;
            }, get_efficient_nested_size(input));
        at::Tensor result = wrap_buffer(output_buffer.reshape(-1),
            new_sizes);
        return result;
      }
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
