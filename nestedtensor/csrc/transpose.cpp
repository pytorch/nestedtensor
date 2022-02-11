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

Tensor _collapse_two_dims(Tensor input, int64_t dim1, int64_t dim2) {
  TORCH_CHECK(dim1 > 0, "dim1: Cannot collapse dim 0.");
  TORCH_CHECK(dim2 > 0, "dim2: Cannot collapse dim 0.");
  TORCH_CHECK(dim2 - 1 == dim1, "dim2 must be one more than dim1.")
  TORCH_CHECK(dim1 == 1 || dim1 == 2, "dim1 must be 1 or 2.")
  TORCH_CHECK(get_dim(input) == 4, "Expected input to be 4 dim.");
  auto input_esizes = get_efficient_nested_size(input);
  Tensor nt_sizes = input_esizes.sizes();

  Tensor sizes_dim1 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
  Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
  Tensor sizes_dim3 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();

  Tensor new_nt_sizes;
  if (dim1 == 1) {
    Tensor collapsed_sizes = sizes_dim1 * sizes_dim2;
    new_nt_sizes = at::cat({collapsed_sizes, sizes_dim3}, 1);
  } else if (dim1 == 2) {
    Tensor collapsed_sizes = sizes_dim2 * sizes_dim3;
    new_nt_sizes = at::cat({sizes_dim1, collapsed_sizes}, 1);
  }
  auto new_esizes = torch::nested_tensor::EfficientSizeNode(input_esizes.structure(), new_nt_sizes);
  Tensor result = wrap_buffer(get_buffer(input), new_esizes);
  TORCH_CHECK(get_dim(result) == 3, "Expected result to be 3 dimensional.");
  return result;

}

template <int grain_size>
std::tuple<at::Tensor, at::Tensor> _create_offsets(Tensor input) {
  TORCH_CHECK(get_dim(input) == 3, "Expected input to be 3 dimensional.");
  Tensor nt_sizes = get_efficient_nested_size(input).sizes();
  int64_t* nt_sizes_ptr = nt_sizes.data_ptr<int64_t>();
  int64_t batch_size = nt_sizes.size(0);
  at::Tensor offsets = torch::empty({1 + batch_size}, torch::kInt32);
  at::Tensor block_offsets = torch::empty({1 + batch_size}, torch::kInt32);
  int* offsets_ptr = offsets.data_ptr<int>();
  int* block_offsets_ptr = block_offsets.data_ptr<int>();
  offsets_ptr[0] = 0;
  block_offsets_ptr[0] = 0;
  int64_t index = 1;
  for (int64_t i = 0; i < batch_size; i++) {
    int64_t size1 = nt_sizes_ptr[i * 2 + 0];
    int64_t size2 = nt_sizes_ptr[i * 2 + 1];
    const int num_chunks_1 = (size1 + grain_size - 1) / grain_size;
    const int num_chunks_2 = (size2 + grain_size - 1) / grain_size;
    offsets_ptr[index] = offsets_ptr[index - 1] + (int)(size1 * size2);
    block_offsets_ptr[index] = block_offsets_ptr[index - 1] + num_chunks_1 * num_chunks_2;
    index++;
  }
  return std::make_tuple(offsets, block_offsets);
}

std::vector<Tensor> _transfer_metadata(std::vector<Tensor> meta_tensors) {
  for (size_t i = 0; i < meta_tensors.size(); i++) {
    meta_tensors[i] = meta_tensors[i].view(-1);
  }
  at::Tensor all_meta = at::cat(meta_tensors);
  all_meta = all_meta.to(at::Device(kCUDA), torch::kInt32, true, true);
  std::vector<Tensor> result_meta_tensors;
  int64_t index = 0;
  for (size_t i = 0; i < meta_tensors.size(); i++) {
    Tensor result_slice = all_meta.narrow(0, index, meta_tensors[i].numel());
    index += meta_tensors[i].numel();
    result_meta_tensors.push_back(result_slice);
  }
  return result_meta_tensors;
}

template <typename scalar_t>
Tensor _transpose_nchw_nhwc(Tensor input, Tensor output) {
#ifdef WITH_CUDA
  Tensor collapsed_input = _collapse_two_dims(input, 2, 3);
  Tensor nt_sizes = get_efficient_nested_size(collapsed_input).sizes();
  Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
  Tensor sizes_dim3 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
  Tensor offsets;
  Tensor block_offsets;
  std::tie(offsets, block_offsets) = _create_offsets<32>(collapsed_input);
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  Tensor input_buffer = get_buffer(input);
  Tensor output_buffer = get_buffer(output);
  TORCH_CHECK(input_buffer.is_cuda(), "Expected input_buffer to be CUDA.");
  TORCH_CHECK(output_buffer.is_cuda(), "Expected output_buffer to be CUDA.");
  int* block_offsets_ptr = block_offsets.data_ptr<int>();
  int batch_size = sizes_dim2.numel();
  int block_numel = block_offsets_ptr[batch_size];
  auto result_meta_tensors = _transfer_metadata({offsets,
                                                 block_offsets});
  nested_tensor::cuda::transpose_nchw_nhwc_kernelLauncher(
      input_buffer.data_ptr<scalar_t>(),
      output_buffer.data_ptr<scalar_t>(),
      result_meta_tensors[1].data_ptr<int>(), // block_offsets
      result_meta_tensors[0].data_ptr<int>(), // offsets
      batch_size, 
      block_numel,
      sizes_dim2[0].item<int64_t>(),
      defaultStream
      );
#endif
  return output;
}

Tensor transpose_nchw_nhwc(Tensor input) {
  TORCH_CHECK(get_dim(input) == 4, "transpose_nchw_nhwc needs 4d input.");
  TORCH_CHECK(get_is_contiguous(input), "transpose_nchw_nhwc input needs to be contiguous.");
  auto input_opt_sizes = get_opt_sizes(input);
  TORCH_CHECK(input_opt_sizes[1], "Expected first dimension to be regular.");
  Tensor input_buffer = get_buffer(input);
  auto new_sizes = map_efficient_size([](int64_t* size_ptr, int64_t size) {
      int64_t tmp = size_ptr[0];
      size_ptr[0] = size_ptr[2];
      size_ptr[2] = tmp;
      tmp = size_ptr[0];
      size_ptr[0] = size_ptr[1];
      size_ptr[1] = tmp;
      }, get_efficient_nested_size(input));
  Tensor output = wrap_buffer(at::empty_like(input_buffer), new_sizes);
  if (get_dtype(input) == torch::kFloat16) {
    return _transpose_nchw_nhwc<c10::Half>(input, output);
  }
  if (get_dtype(input) == torch::kFloat) {
    return _transpose_nchw_nhwc<float>(input, output);
  }
  TORCH_CHECK(false, "Given dtype ", get_dtype(input), " not supported.");
}

template <typename scalar_t>
Tensor _transpose_nhwc_nchw(Tensor input, Tensor output) {
#ifdef WITH_CUDA
  Tensor collapsed_input = _collapse_two_dims(input, 1, 2);
  Tensor nt_sizes = get_efficient_nested_size(collapsed_input).sizes();
  Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
  Tensor sizes_dim3 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
  Tensor offsets;
  Tensor block_offsets;
  std::tie(offsets, block_offsets) = _create_offsets<32>(collapsed_input);
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  Tensor input_buffer = get_buffer(input);
  Tensor output_buffer = get_buffer(output);
  int* block_offsets_ptr = block_offsets.data_ptr<int>();
  int batch_size = sizes_dim3.numel();
  int block_numel = block_offsets_ptr[batch_size];
  auto result_meta_tensors = _transfer_metadata({offsets,
                                                 block_offsets});
  nested_tensor::cuda::transpose_nhwc_nchw_kernelLauncher(
      input_buffer.data_ptr<scalar_t>(),
      output_buffer.data_ptr<scalar_t>(),
      result_meta_tensors[1].data_ptr<int>(), // block_offsets
      result_meta_tensors[0].data_ptr<int>(), // offsets
      batch_size, 
      block_numel,
      sizes_dim3[0].item<int64_t>(),
      defaultStream
      );
#endif
  return output;
}

Tensor transpose_nhwc_nchw(Tensor input) {
  TORCH_CHECK(get_dim(input) == 4, "transpose_nhwc_nchw needs 4d input.");
  TORCH_CHECK(get_is_contiguous(input), "transpose_nhwc_nchw input needs to be contiguous.");
  auto input_opt_sizes = get_opt_sizes(input);
  TORCH_CHECK(input_opt_sizes[3], "Expected last dimension to be regular.");
  Tensor input_buffer = get_buffer(input);
  auto new_sizes = map_efficient_size([](int64_t* size_ptr, int64_t size) {
      // nhwc
      int64_t tmp = size_ptr[0];
      size_ptr[0] = size_ptr[2];
      size_ptr[2] = tmp;
      // ncwh
      tmp = size_ptr[1];
      size_ptr[1] = size_ptr[2];
      size_ptr[2] = tmp;
      // nchw
      }, get_efficient_nested_size(input));
  Tensor output = wrap_buffer(at::empty_like(input_buffer), new_sizes);
  if (get_dtype(input) == torch::kFloat16) {
    return _transpose_nhwc_nchw<c10::Half>(input, output);
  }
  if (get_dtype(input) == torch::kFloat) {
    return _transpose_nhwc_nchw<float>(input, output);
  }
  TORCH_CHECK(false, "Given dtype ", get_dtype(input), " not supported.");
}

}
