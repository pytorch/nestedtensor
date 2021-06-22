#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

__global__
void transpose(
    c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grain_size = 256;
  const int tid = threadIdx.x;
  const int size2 = sizes_dim2[batch_id];
  const int size3 = sizes_dim3[batch_id];
  const int num_chunks = (size2 * size3) / grain_size;
  for (int id = 0; id < num_chunks; id++) {
    int ii = id * grain_size + tid;
  // for (int ii = 0; ii < size2 * size3; ii++) {
    const int j = (ii % size3) * size2;
    const int i = (ii / size3);
    output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
  // }
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < (size2 * size3)) {
    int ii = leftover + tid;
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
  grid.x = batch_size;

  transpose<<<grid, 256, 0, stream>>>(
      input,
      output,
      offsets,
      sizes_dim2,
      sizes_dim3,
      batch_size);
}

}
} // namespace nested_tensor
