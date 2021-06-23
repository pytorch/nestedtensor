#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<int num_threads_sqrt>
__global__
void transpose(
    c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* blocks2,
    const int* blocks3,
    const int* blocks_batch_dim,
    const int* size_dim2,
    const int* size_dim3)
{
  __shared__ c10::Half tile[num_threads_sqrt][num_threads_sqrt + 1];
  const int block_id  = blockIdx.x;
  const int batch_id = blocks_batch_dim[block_id];
  const int grain_size = num_threads_sqrt;
  const int tid2 = threadIdx.x;
  const int tid3 = threadIdx.y;
  const int id2 = blocks2[block_id];
  const int id3 = blocks3[block_id];
  const int size2 = size_dim2[batch_id];
  const int size3 = size_dim3[batch_id];
  const int offset = offsets[batch_id];

  for (int bindx = 0; bindx < 4; bindx++) {
    const int ii2 = id2 + tid2;
    const int ii3 = id3 + tid3 + 8 * bindx;
    if (ii2 < size2 && ii3 < size3) {
      const int ii = ii2 * size3 + ii3;
      tile[tid2][tid3 + 8 * bindx] = __ldg(reinterpret_cast<const __half*>(input) + offset + ii);
    }
  }
  for (int bindx = 0; bindx < 4; bindx++) {
    const int ii2 = id2 + tid2;
    const int ii3 = id3 + tid3 + 8 * bindx;
    if (ii2 < size2 && ii3 < size3) {
      const int ii21 = id2 + tid2;
      const int ii31 = id3 + tid3 + 8 * bindx;
      const int ii1 = ii21 * size3 + ii31;
      const int j = (ii1 % size3) * size2;
      const int i = (ii1 / size3);
      output[offset + j + i] = tile[tid2][tid3 + 8 * bindx];
    }
  }
}

void transpose_kernelLauncher(
    c10::Half* input, // [batch_size x None]
    c10::Half* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets,
    const int* blocks2,
    const int* blocks3,
    const int* blocks_batch_dim,
    const int* size_dim2,
    const int* size_dim3,
    const int block_numel,
    const int numel,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = block_numel;

  transpose<32><<<grid, dim3(32, 8), 0, stream>>>(
      input,
      output,
      offsets,
      blocks2,
      blocks3,
      blocks_batch_dim,
      size_dim2,
      size_dim3);
}

}
} // namespace nested_tensor
