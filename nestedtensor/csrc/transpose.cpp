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
#include <nestedtensor/csrc/transpose.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor transpose_buffer(Tensor nt_sizes_, Tensor input_buffer, Tensor output_buffer) {
#ifdef WITH_CUDA
  Tensor sizes_dim2 = at::native::narrow(nt_sizes_, 1, 0, 1).contiguous();
  Tensor sizes_dim3 = at::native::narrow(nt_sizes_, 1, 1, 1).contiguous();
  Tensor nt_sizes_all = (sizes_dim2 * sizes_dim3).to(torch::kInt32);
  int* nt_sizes_all_ptr = nt_sizes_all.data_ptr<int>();
  int64_t* sizes_dim2_ptr = sizes_dim2.data_ptr<int64_t>();
  int64_t* sizes_dim3_ptr = sizes_dim3.data_ptr<int64_t>();
  int64_t batch_size = nt_sizes_.size(0);
  // int64_t input_buffer_numel = input_buffer.numel();
  at::Tensor offsets = torch::zeros({1 + batch_size}, torch::kInt32);
  int* offsets_ptr = offsets.data_ptr<int>();
  at::Tensor block_offsets = torch::zeros({1 + batch_size}, torch::kInt32);
  int* block_offsets_ptr = block_offsets.data_ptr<int>();
  int64_t index = 1;
  int grain_size = 32;
  for (int64_t i = 0; i < batch_size; i++) {
    const int size2 = sizes_dim2_ptr[i];
    const int size3 = sizes_dim3_ptr[i];
    const int num_chunks_2 = (size2 + grain_size - 1) / grain_size;
    const int num_chunks_3 = (size3 + grain_size - 1) / grain_size;
    offsets_ptr[index] = offsets_ptr[index - 1] + (int)(nt_sizes_all_ptr[i]);
    block_offsets_ptr[index] = block_offsets_ptr[index - 1] + num_chunks_2 * num_chunks_3; 
    index++;
  }
  int block_numel = block_offsets_ptr[batch_size];
  sizes_dim2 = sizes_dim2.reshape(-1);
  sizes_dim3 = sizes_dim3.reshape(-1);

  at::Tensor all_meta = at::cat({offsets, block_offsets, sizes_dim2, sizes_dim3});

  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  all_meta = all_meta.to(at::Device(kCUDA), torch::kInt32, true, true);
  index = 0;
  offsets = all_meta.narrow(0, 0, offsets.numel());
  index += offsets.numel();
  block_offsets = all_meta.narrow(0, index, block_offsets.numel());
  index += block_offsets.numel();
  sizes_dim2 = all_meta.narrow(0, index, sizes_dim2.size(0));
  index += sizes_dim2.size(0);
  sizes_dim3 = all_meta.narrow(0, index, sizes_dim3.size(0));

  c10::Half* input_ptr = input_buffer.data_ptr<c10::Half>();
  c10::Half* output_ptr = output_buffer.data_ptr<c10::Half>();

  // std::cout << "at::cuda::warp_size(): " << at::cuda::warp_size() << std::endl;
  nested_tensor::cuda::transpose_kernelLauncher(
      input_ptr,
      output_ptr,
      block_offsets.data_ptr<int>(),
      offsets.data_ptr<int>(),
      batch_size, 
      block_numel,
      sizes_dim2.data_ptr<int>(),
      sizes_dim3.data_ptr<int>(),
      defaultStream
      );
  return output_buffer.reshape(-1);
#endif
  TORCH_CHECK(false, "transpose_buffer needs CUDA.");
}

Tensor transpose_nhwc_nchw_out(Tensor input, Tensor output) {
#ifdef WITH_CUDA
  TORCH_CHECK(get_dim(input) == 4, "transpose_nhwc_nchw_out needs 4d input.");
  TORCH_CHECK(get_is_channel_last(input), "transpose_nhwc_nchw_out input needs to be channel last.");
  TORCH_CHECK(get_dim(output) == 4, "transpose_nhwc_nchw_out needs 4d output.");
  TORCH_CHECK(get_is_contiguous(output), "transpose_nhwc_nchw_out output needs to be contiguous.");
  Tensor nt_sizes = get_efficient_nested_size(input).sizes();
  Tensor nt_sizes_0 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
  Tensor nt_sizes_1 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
  Tensor nt_sizes_2 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();
  Tensor nt_sizes_1_2 = nt_sizes_1 * nt_sizes_2;
  nt_sizes = at::cat({nt_sizes_1_2, nt_sizes_0}, 1);
  Tensor input_buffer = get_buffer_channel_last(input);
  std::cout << "0 input_buffer.sizes(): " << input_buffer.sizes() << std::endl;
  std::cout << "0 input_buffer.strides(): " << input_buffer.strides() << std::endl;
  Tensor output_buffer = get_buffer(output);
  output_buffer = transpose_buffer(nt_sizes, input_buffer, output_buffer);
  output_buffer = output_buffer.reshape(-1);
  return wrap_buffer(std::move(output_buffer), get_efficient_nested_size(input));
#endif
  TORCH_CHECK(false, "transpose_nhwc_nchw needs CUDA.");
}

Tensor transpose_nhwc_nchw(Tensor input) {
  TORCH_CHECK(get_dim(input) == 4, "transpose_nhwc_nchw needs 4d input.");
  TORCH_CHECK(get_is_channel_last(input), "transpose_nhwc_nchw input needs to be channel last.");
  Tensor input_buffer = get_buffer_channel_last(input);
  Tensor output = wrap_buffer(at::empty_like(input_buffer),
                              get_efficient_nested_size(input));
  return transpose_nhwc_nchw_out(input, output);
}

// TODO: Might actually return nwhc (same for inverse above), but for our applications this doesn't matter.
Tensor transpose_nchw_nhwc_out(Tensor input, Tensor output) {
#ifdef WITH_CUDA
  TORCH_CHECK(get_dim(input) == 4, "transpose_nchw_nhwc_out needs 4d input.");
  TORCH_CHECK(get_is_contiguous(input), "transpose_nchw_nhwc_out input needs to be contiguous.");
  TORCH_CHECK(get_dim(output) == 4, "transpose_nchw_nhwc_out needs 4d output.");
  TORCH_CHECK(get_is_channel_last(output), "transpose_nchw_nhwc_out output needs to be channel last.");
  Tensor nt_sizes =
      get_efficient_nested_size(input).sizes();
  Tensor nt_sizes_0 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
  Tensor nt_sizes_1 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
  Tensor nt_sizes_2 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();
  Tensor nt_sizes_1_2 = nt_sizes_1 * nt_sizes_2;
  nt_sizes = at::cat({nt_sizes_0, nt_sizes_1_2}, 1);
  Tensor input_buffer = get_buffer(input);
  std::cout << "0 input_buffer.sizes(): " << input_buffer.sizes() << std::endl;
  std::cout << "0 input_buffer.strides(): " << input_buffer.strides() << std::endl;
  Tensor output_buffer = at::empty_like(input_buffer);
  output_buffer = transpose_buffer(nt_sizes, input_buffer, output_buffer);
  output_buffer = output_buffer.reshape(-1);
  return wrap_buffer_channel_last(std::move(output_buffer), get_efficient_nested_size(input));
#endif
  TORCH_CHECK(false, "transpose_nchw_nhwc needs CUDA.");
}

Tensor transpose_nchw_nhwc(Tensor input) {
  TORCH_CHECK(get_dim(input) == 4, "transpose_nchw_nhwc needs 4d input.");
  TORCH_CHECK(get_is_contiguous(input), "transpose_nchw_nhwc input needs to be contiguous.");
  Tensor input_buffer = get_buffer(input);
  Tensor output = wrap_buffer_channel_last(at::empty_like(input_buffer),
                                           get_efficient_nested_size(input));
  return transpose_nchw_nhwc_out(input, output);
}
}
