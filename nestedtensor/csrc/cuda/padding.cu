#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/attention.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<typename T>
__global__
void add_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int batch_size,
    const int output_stride,
    const int inner_size) 
{
  const int batch_id  = blockIdx.x;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]) * inner_size;
  const int num_chunks = range / grain_size;
  for (int id = 0; id < num_chunks; id++) {
    output[batch_id * output_stride + id * grain_size + tid]
      = input[offsets[batch_id] * inner_size + id * grain_size + tid];
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < range) {
    output[batch_id * output_stride + leftover + tid]
      = input[offsets[batch_id] * inner_size + leftover + tid];
  }
}

template<typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets, // [batch_size]
    const int batch_size,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  add_padding<float><<<grid, 1024, 0, stream>>>(
      input,
      output,
      offsets,
      batch_size,
      output_stride,
      inner_size);
}

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    const int* offsets,
    const int batch_size,
    const int output_stride,
    const int inner_size,
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
    const int batch_size,
    const int output_stride,
    const int inner_size)
{
  const int batch_id  = blockIdx.x;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]) * inner_size;
  const int num_chunks = range / grain_size;
  for (int id = 0; id < num_chunks; id++) {
    output[offsets[batch_id] * inner_size + id * grain_size + tid]
     = input[batch_id * output_stride + id * grain_size + tid];
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < range) {
    output[offsets[batch_id] * inner_size + leftover + tid]
     = input[batch_id * output_stride + leftover + tid];
  }
}

template<typename T>
void remove_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets, // [batch_size]
    const int batch_size,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  remove_padding<float><<<grid, 1024, 0, stream>>>(
      input,
      output,
      offsets,
      batch_size,
      output_stride,
      inner_size);
}

template void remove_padding_kernelLauncher<float>(
    float* input,
    float* output,
    const int* offsets,
    const int batch_size,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream);
}
}
