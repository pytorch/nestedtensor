#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<int unroll_chunks, int block_mult, int num_threads>
__global__
void transpose(
    c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int batch_size)
{
  const int batch_id  = blockIdx.x / block_mult;
  const int block_id = blockIdx.x % block_mult;
  const int grain_size = num_threads * block_mult;
  const int tid = threadIdx.x;
  const int size2 = sizes_dim2[batch_id];
  const int size3 = sizes_dim3[batch_id];
  const int num_chunks = (size2 * size3) / grain_size;
  const int num_4_chunks = num_chunks / unroll_chunks;
  for (int id4 = 0; id4 < num_4_chunks; id4++) {
#pragma unroll
    for (int id = 0; id < unroll_chunks; id++) {
      int ii = (id4 * unroll_chunks + id) * grain_size + (block_id) * num_threads + tid;
      const int j = (ii % size3) * size2;
      const int i = (ii / size3);
      output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
    }
  }
  for (int id = num_4_chunks * unroll_chunks; id < num_chunks; id++) {
    int ii = id * grain_size + (block_id) * num_threads + tid;
    const int j = (ii % size3) * size2;
    const int i = (ii / size3);
    output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + (block_id) * num_threads + tid < (size2 * size3)) {
    int ii = leftover + (block_id) * num_threads + tid;
    const int j = (ii % size3) * size2;
    const int i = (ii / size3);
    output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
  }
}

void transpose_kernelLauncher(
    c10::Half* input, // [batch_size x None]
    c10::Half* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets, // [batch_size]
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size * 512;

  transpose<8, 512, 512><<<grid, 512, 0, stream>>>(
      input,
      output,
      offsets,
      sizes_dim2,
      sizes_dim3,
      batch_size);
}

}
} // namespace nested_tensor
