#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <c10/util/Half.h>
#endif
#include <nestedtensor/csrc/masking.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

#ifdef WITH_CUDA
Tensor transpose_buffer(Tensor nt_sizes_, Tensor input_buffer, Tensor output_buffer) {
  Tensor sizes_dim2 = at::native::narrow(nt_sizes_, 1, 0, 1).contiguous();
  Tensor sizes_dim3 = at::native::narrow(nt_sizes_, 1, 1, 1).contiguous();
  Tensor nt_sizes_all = (sizes_dim2 * sizes_dim3).to(torch::kInt32);
  int* nt_sizes_all_ptr = nt_sizes_all.data_ptr<int>();
  int64_t* sizes_dim2_ptr = sizes_dim2.data_ptr<int64_t>();
  int64_t* sizes_dim3_ptr = sizes_dim3.data_ptr<int64_t>();
  int64_t batch_size = nt_sizes_.size(0);
  int64_t input_buffer_numel = input_buffer.numel();
  at::Tensor offsets = torch::zeros({1 + batch_size}, torch::kInt32);
  int* offsets_ptr = offsets.data_ptr<int>();
  int64_t index = 1;
  int grain_size = 32;
  std::vector<int> blocks2_vec;
  blocks2_vec.reserve(input_buffer_numel / (grain_size * grain_size));
  std::vector<int> blocks3_vec;
  blocks3_vec.reserve(input_buffer_numel / (grain_size * grain_size));
  std::vector<int> blocks_batch_dim_vec;
  blocks_batch_dim_vec.reserve(input_buffer_numel / (grain_size * grain_size));
  for (int64_t i = 0; i < batch_size; i++) {
    const int size2 = sizes_dim2_ptr[i];
    const int size3 = sizes_dim3_ptr[i];
    const int num_chunks_2 = (size2 + grain_size - 1) / grain_size;
    const int num_chunks_3 = (size3 + grain_size - 1) / grain_size;
    offsets_ptr[index] = offsets_ptr[index - 1] + (int)(nt_sizes_all_ptr[i]);
    for (int id2 = 0; id2 < num_chunks_2; id2++) {
      for (int id3 = 0; id3 < num_chunks_3; id3++) {
        blocks2_vec.push_back(id2 * grain_size);
        blocks3_vec.push_back(id3 * grain_size);
        blocks_batch_dim_vec.push_back(i);
      }
    }
    index++;
  }
  at::Tensor blocks2 = torch::tensor(blocks2_vec);
  at::Tensor blocks3 = torch::tensor(blocks3_vec);
  at::Tensor blocks_batch_dim = torch::tensor(blocks_batch_dim_vec);
  sizes_dim2 = sizes_dim2.reshape(-1);
  sizes_dim3 = sizes_dim3.reshape(-1);

  at::Tensor all_meta = at::cat({offsets, blocks2, blocks3, blocks_batch_dim, sizes_dim2, sizes_dim3});

  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  all_meta = all_meta.to(at::Device(kCUDA), torch::kInt32, true, true);
  std::vector<int64_t> split_sizes;
  split_sizes.push_back(offsets.numel());
  split_sizes.push_back(blocks2_vec.size());
  split_sizes.push_back(blocks3_vec.size());
  split_sizes.push_back(blocks_batch_dim_vec.size());
  split_sizes.push_back(sizes_dim2.size(0));
  split_sizes.push_back(sizes_dim3.size(0));
  std::vector<at::Tensor> split_all_meta = at::split_with_sizes(all_meta, c10::IntArrayRef(split_sizes), 0);
  offsets = split_all_meta[0];
  blocks2 = split_all_meta[1];
  blocks3 = split_all_meta[2];
  blocks_batch_dim = split_all_meta[3];
  sizes_dim2 = split_all_meta[4];
  sizes_dim3 = split_all_meta[5];

  c10::Half* input_ptr = input_buffer.data_ptr<c10::Half>();
  c10::Half* output_ptr = output_buffer.data_ptr<c10::Half>();
  nested_tensor::cuda::transpose_kernelLauncher(
      input_ptr,
      output_ptr,
      offsets.data_ptr<int>(),
      blocks2.data_ptr<int>(),
      blocks3.data_ptr<int>(),
      blocks_batch_dim.data_ptr<int>(),
      sizes_dim2.data_ptr<int>(),
      sizes_dim3.data_ptr<int>(),
      blocks2_vec.size(),
      input_buffer_numel,
      defaultStream
      );
  return output_buffer.reshape(-1);
}
#endif

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
        Tensor nt_sizes =
            get_efficient_nested_size(input).sizes();
        Tensor nt_sizes_0 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
        Tensor nt_sizes_1 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
        Tensor nt_sizes_2 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();
        Tensor nt_sizes_1_2 = nt_sizes_1 * nt_sizes_2;
        nt_sizes = at::cat({nt_sizes_0, nt_sizes_1_2}, 1);
        Tensor input_buffer = get_buffer(input);
        Tensor output_buffer = input_buffer.clone();
        output_buffer = transpose_buffer(nt_sizes, input_buffer, output_buffer);
        output_buffer = output_buffer.reshape({-1, weight.size(1)});
        at::Tensor result_buffer = at::matmul(output_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        nt_sizes_0.fill_(weight_size_0);
        nt_sizes = at::cat({nt_sizes_1_2, nt_sizes_0}, 1);
        output_buffer.resize_as_(result_buffer);
        output_buffer = transpose_buffer(nt_sizes,
                                         result_buffer.reshape(-1),
                                         output_buffer.reshape(-1));

        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[0] = weight_size_0;
            }, get_efficient_nested_size(input));
        return wrap_buffer(output_buffer.reshape(-1), new_sizes);
      }
    }
  }
#endif
  if (groups == 1) {
    std::cout << "Running" << std::endl;
    // std::cout << "weight.sizes(): " << weight.sizes();
    // std::cout << " stride: " << stride;
    // std::cout << " padding: " << padding;
    // std::cout << " dilation: " << dilation;
    // std::cout << " groups: " << groups;
    std::cout << "1" << std::endl;
    at::Tensor data;
    std::cout << "2" << std::endl;
    at::Tensor mask;
    std::cout << "3" << std::endl;
    std::tie(data, mask) = to_tensor_mask(input, 4);
    std::cout << "4" << std::endl;
    // std::cout << "data: " << data << std::endl;
    // std::cout << "mask: " << mask << std::endl;
    at::Tensor result_data = at::conv2d(data, weight, bias, stride, padding, dilation, groups);
    std::cout << "5" << std::endl;
    mask = mask.to(data.dtype());
    std::cout << "6" << std::endl;
    at::Tensor result_mask = at::conv2d(mask, weight, bias, stride, padding, dilation, groups);
    std::cout << "7" << std::endl;
    // int64_t mask_weight_numel = weight.size(1) * weight.size(2) * weight.size(3);
    at::Tensor mask_weight_numel_tensor = at::max(result_mask);
    std::cout << "8" << std::endl;
    // std::cout << "mask_weight_numel_tensor: " << mask_weight_numel_tensor << std::endl;
    // std::cout << "result_mask: " << result_mask << std::endl;
    at::Tensor result_mask2 = result_mask.eq(mask_weight_numel_tensor);
    std::cout << "9" << std::endl;
    // std::cout << "result_data: " << result_data << std::endl;
    // std::cout << "result_mask2: " << result_mask2 << std::endl;
    auto opt_result = nt_from_tensor_mask(result_data, result_mask2, 1);
    std::cout << "0" << std::endl;
    TORCH_CHECK(opt_result, "Result does not form a valid NestedTensor.");
    std::cout << "Done Running" << std::endl;
    return *opt_result;
  }
  if (bias) {
      // std::cout << " (*bias).sizes(): " << (*bias).sizes();
      // std::cout << std::endl;
      return map_nested_tensor(
          [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
            return at::conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
          },
          input,
          weight,
          *bias);
  }
  // std::cout << std::endl;
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
