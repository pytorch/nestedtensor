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
    const int* lengths,
    const int batch_size,
    const int output_stride) 
{
  const int batch_id  = blockIdx.x;
  for (int i = 0; i < lengths[batch_id + 1] - lengths[batch_id]; i++) {
    output[batch_id * output_stride + i] = input[lengths[batch_id] + i];
  }
}

template<typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1))]
    const int* lengths, // [batch_size]
    int batch_size,
    int output_stride,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  add_padding<float><<<grid, 1, 0, stream>>>(
      input,
      output,
      lengths,
      batch_size,
      output_stride);
}

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    const int* lengths,
    int batch_size,
    int output_stride,
    const cudaStream_t stream);
}
}
