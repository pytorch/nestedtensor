#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/padding.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<typename T>
__global__
void add_padding_1(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size)
{
  const int batch_id  = blockIdx.y;
  const int grid_id  = blockIdx.x;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int batch_input_offset = offsets[batch_id];
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int numel_i = sizes_i[0];
  const int batch_output_offset = batch_id * output_sizes[1];
  for (int ii = 0; ii < (output_sizes[1] / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int output_offset = batch_output_offset + i;
    if (i < sizes_i[0]) {
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
  const int i = (output_sizes[1] / grainsize) * grainsize + tid;
  if (i < output_sizes[1]) {
    const int output_offset = batch_output_offset + i;
    if (i < sizes_i[0]) {
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
}

template<typename T>
__global__
void add_padding_2(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size)
{
  const int batch_id  = blockIdx.y;
  const int grid_id  = blockIdx.x;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int offset = offsets[batch_id];
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  const int output_offset = batch_id * output_sizes[1] * output_sizes[2];
  const int output_numel = output_sizes[1] * output_sizes[2];
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes[2]);
    const int i1 = i % output_sizes[2];
    if (i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes[2]);
    const int i1 = i % output_sizes[2];
    if (i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template<typename T, int grainsize>
__global__
void add_padding_3(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_3,
    int output_sizes_2_3,
    int output_numel) 
{
  const int batch_id  = blockIdx.x;
  const int i0 = blockIdx.y;
  const int tid = threadIdx.x;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int sizes_0 = sizes_i[0];
  int i = tid;
  output = output + batch_id * output_numel + i0 * output_sizes_2_3;
  if (i0 < sizes_0) {
    const int sizes_1 = sizes_i[1];
    const int sizes_2 = sizes_i[2];
    const int sizes_1_2 = sizes_1 * sizes_2;
    input = input + offsets[batch_id] + i0 * sizes_1_2;
    bool valid_0 = i0 < sizes_0;
    for (;i < output_sizes_2_3;) {
      const int i1 = i / output_sizes_3;
      const int i2 = i % output_sizes_3;
      const bool valid = i1 < sizes_1 && i2 < sizes_2;
      const int input_offset = valid ? i1 * sizes_2 + i2 : 0;
      output[i] = valid ? input[input_offset] : padding_value;
      i += grainsize;
    }
  } else {
    for (;i < output_sizes_2_3;) {
      output[i] = padding_value;
      i += grainsize;
    }
  }
}

template<typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;
  grid.y = output_sizes[1];
  if (input_dim == 1) {
    add_padding_1<T><<<grid, 256, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes,
        batch_size);
  }
  if (input_dim == 2) {
    add_padding_2<T><<<grid, 256, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes,
        batch_size);
  }
  if (input_dim == 3) {
    add_padding_3<T, 256><<<grid, 256, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[3],
        output_sizes[2] * output_sizes[3],
        output_sizes[1] * output_sizes[2] * output_sizes[3]);
  }
}

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    float padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream);

template void add_padding_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    c10::Half padding_value,
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
