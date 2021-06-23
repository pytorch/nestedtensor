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

// TODO: Determine chunks and map from batch_id to tile id
Tensor transpose_buffer(Tensor nt_sizes_, Tensor input_buffer, Tensor output_buffer) {
  Tensor nt_sizes_0 = at::native::narrow(nt_sizes_, 1, 0, 1).contiguous();
  Tensor nt_sizes_1_2 = at::native::narrow(nt_sizes_, 1, 1, 1).contiguous();
  // std::cout << "nt_sizes_0: " << nt_sizes_0 << std::endl;
  // std::cout << "nt_sizes_1_2: " << nt_sizes_1_2 << std::endl;
  Tensor nt_sizes_all = nt_sizes_0 * nt_sizes_1_2;
  int64_t* nt_sizes_all_ptr = nt_sizes_all.data_ptr<int64_t>();
  int64_t* sizes_dim2_ptr = nt_sizes_0.data_ptr<int64_t>();
  int64_t* sizes_dim3_ptr = nt_sizes_1_2.data_ptr<int64_t>();
  std::vector<int> offsets_vec;
  offsets_vec.reserve(1 + nt_sizes_all.size(0));
  offsets_vec.push_back(0);
  int64_t index = 1;
  int grain_size = 16;
  std::vector<int> blocks2_vec;
  std::vector<int> blocks3_vec;
  std::vector<int> blocks_batch_dim_vec;
  std::vector<int> sizes_dim2_vec;
  std::vector<int> sizes_dim3_vec;
  for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
    const int size2 = sizes_dim2_ptr[i];
    const int size3 = sizes_dim3_ptr[i];
    const int num_chunks_2 = (size2 + grain_size - 1) / grain_size;
    const int num_chunks_3 = (size3 + grain_size - 1) / grain_size;
    offsets_vec.push_back(offsets_vec[index - 1] + (int)(nt_sizes_all_ptr[i]));
    for (int id2 = 0; id2 < num_chunks_2; id2++) {
      for (int id3 = 0; id3 < num_chunks_3; id3++) {
        blocks2_vec.push_back(id2 * grain_size);
        blocks3_vec.push_back(id3 * grain_size);
        blocks_batch_dim_vec.push_back(i);
      }
    }
    sizes_dim2_vec.push_back(size2);
    sizes_dim3_vec.push_back(size3);
    index++;
  }
  // std::cout << "offsets_vec.size(): " << offsets_vec.size() << std::endl;
  at::Tensor offsets = torch::tensor(offsets_vec);
  at::Tensor blocks2 = torch::tensor(blocks2_vec);
  at::Tensor blocks3 = torch::tensor(blocks3_vec);
  at::Tensor blocks_batch_dim = torch::tensor(blocks_batch_dim_vec);
  at::Tensor sizes_dim2 = torch::tensor(sizes_dim2_vec);
  at::Tensor sizes_dim3 = torch::tensor(sizes_dim3_vec);

  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  offsets = offsets.to(at::Device(kCUDA), torch::kInt32, true, true);
  blocks2 = blocks2.to(at::Device(kCUDA), torch::kInt32, true, true);
  blocks3 = blocks3.to(at::Device(kCUDA), torch::kInt32, true, true);
  blocks_batch_dim = blocks_batch_dim.to(at::Device(kCUDA), torch::kInt32, true, true);
  sizes_dim2 = sizes_dim2.to(at::Device(kCUDA), torch::kInt32, true, true);
  sizes_dim3 = sizes_dim3.to(at::Device(kCUDA), torch::kInt32, true, true);


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
      input_buffer.numel(),
      defaultStream
      );
  return output_buffer.reshape(-1);
}

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
        Tensor nt_sizes =
            get_efficient_nested_size(input).sizes();
        // std::cout << "nt_sizes: " << nt_sizes << std::endl;
        Tensor nt_sizes_0 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
        Tensor nt_sizes_1 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
        Tensor nt_sizes_2 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();
        Tensor nt_sizes_1_2 = nt_sizes_1 * nt_sizes_2;
        nt_sizes = at::cat({nt_sizes_0, nt_sizes_1_2}, 1);
        // std::cout << "nt_sizes 0: " << nt_sizes << std::endl;
        Tensor input_buffer = get_buffer(input);
        Tensor output_buffer = input_buffer.clone();
        output_buffer = transpose_buffer(nt_sizes, input_buffer, output_buffer);
        output_buffer = output_buffer.reshape({-1, weight.size(1)});
        // std::cout << "output_buffer.sizes(): " << output_buffer.sizes() << std::endl;
        // std::cout << "weight.sizes(): " << weight.sizes() << std::endl;
        at::Tensor result_buffer = at::matmul(output_buffer, 
            weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
        int64_t weight_size_0 = weight.size(0);
        nt_sizes_0.fill_(weight_size_0);
        nt_sizes = at::cat({nt_sizes_1_2, nt_sizes_0}, 1);
        output_buffer.resize_as_(result_buffer);
        // std::cout << "nt_sizes 1: " << nt_sizes << std::endl;
        output_buffer = transpose_buffer(nt_sizes,
                                         result_buffer.reshape(-1),
                                         output_buffer.reshape(-1));

        auto new_sizes = map_efficient_size([&weight_size_0](int64_t* size_ptr, int64_t size) {
            size_ptr[0] = weight_size_0;
            }, get_efficient_nested_size(input));
        // std::cout << "new_sizes.sizes(): " << new_sizes.sizes() << std::endl;
        return wrap_buffer(output_buffer.reshape(-1), new_sizes);
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
