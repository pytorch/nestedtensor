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
  for (int i = 0; i < (offsets[batch_id + 1] - offsets[batch_id]) * inner_size; i++) {
    output[batch_id * output_stride + i] = input[offsets[batch_id] * inner_size + i];
  }
}

template<typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1))]
    const int* offsets, // [batch_size]
    const int batch_size,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  add_padding<float><<<grid, 1, 0, stream>>>(
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
}
}
