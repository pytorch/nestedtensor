#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/padding.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<typename T>
__global__
void add_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grid_id  = blockIdx.y;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int offset = offsets[batch_id];
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
  int output_offset = batch_id * output_sizes[0] * output_sizes[1] * output_sizes[2];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * output_sizes[1] * output_sizes[2];
    const int i1_offset = i1 * output_sizes[2];
    output[output_offset + i0_offset + i1_offset + i2] = input[offset + i];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * output_sizes[1] * output_sizes[2];
    const int i1_offset = i1 * output_sizes[2];
    output[output_offset + i0_offset + i1_offset + i2] = input[offset + i];
  }
}

template<typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;
  grid.y = 16;

  add_padding<T><<<grid, 256, 0, stream>>>(
      input,
      output,
      offsets,
      input_sizes,
      input_dim,
      output_sizes,
      batch_size);
}

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream);

template void add_padding_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream);

template<typename T>
__global__
void add_padding_mask(
    const T* input,
    T* output,
    int* output_mask,
    const int* offsets,
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size)
{
  const int batch_id  = blockIdx.x;
  for (int i = 0; i < (offsets[batch_id + 1] - offsets[batch_id]); i++) {
    output_mask[batch_id*mask_stride + i] = 1;
  }
  for (int i = 0; i < (offsets[batch_id + 1] - offsets[batch_id]) * inner_size; i++) {
    output[batch_id * output_stride + i] = input[offsets[batch_id] * inner_size + i];
  }
}

template<typename T>
void add_padding_mask_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    int* output_mask, // [batch_size x max(input.nested_size(1))]
    const int* offsets, // [batch_size]
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  add_padding_mask<float><<<grid, 1, 0, stream>>>(
      input,
      output,
      output_mask,
      offsets,
      batch_size,
      mask_stride,
      output_stride,
      inner_size);
}

template void add_padding_mask_kernelLauncher<float>(
    float* input,
    float* output,
    int* output_mask,
    const int* offsets,
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream);

template<typename T>
__global__
void remove_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grid_id  = blockIdx.y;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
  int input_offset = batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
}

template<typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;
  grid.y = 16;

  remove_padding<T><<<grid, 256, 0, stream>>>(
    input,
    output,
    offsets,
    input_sizes,
    output_sizes,
    output_dim,
    batch_size);
}

template void remove_padding_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream);

template void remove_padding_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream);
}
}
