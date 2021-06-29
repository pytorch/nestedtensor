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
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int* size_dim2,
    const int* size_dim3)
{
  __shared__ c10::Half tile[num_threads_sqrt][num_threads_sqrt + 1];
  __shared__ int batch_id;
  const int block_id  = blockIdx.x;
  const int tid2 = threadIdx.x;
  const int tid3 = threadIdx.y;
  int batch_id_search = tid2 * 32 + tid3;
  while (batch_id_search < batch_size) {
    if (block_offsets[batch_id_search] <= block_id && 
        block_id < block_offsets[batch_id_search + 1]) {
      batch_id = batch_id_search;
    }
    batch_id_search += 256;
  }
  __syncthreads();
  const int grain_size = num_threads_sqrt;
  const int size2 = size_dim2[batch_id];
  const int size3 = size_dim3[batch_id];
  const int block_offset = block_offsets[batch_id];
  const int offset = offsets[batch_id];

  const int num_chunks_3 = (size3  + grain_size - 1) / grain_size;
  const int current_block = block_id - block_offset;
  const int ii3 = (current_block % num_chunks_3) * grain_size + tid3;
  for (int sub = 0; sub < 4; sub++) {
    const int ii2 = (current_block / num_chunks_3) * grain_size + tid2 + sub * 8;
    if (ii2 < size2 && ii3 < size3) {
      const int ii = ii2 * size3 + ii3;
      tile[tid2 + sub * 8][tid3] = __ldg(reinterpret_cast<const __half*>(input) + offset + ii);
    }
  }

  __syncthreads();

  const int ii21 = (current_block / num_chunks_3) * grain_size + tid3;
  for (int sub = 0; sub < 4; sub++) {
    const int ii31 = (current_block % num_chunks_3) * grain_size + tid2 + sub * 8;
    if (ii21 < size2 && ii31 < size3) {
      const int ii1 = ii21 * size3 + ii31;
      const int j = (ii1 % size3) * size2;
      const int i = (ii1 / size3);
      output[offset + j + i] = tile[tid3][tid2 + sub * 8];
    }
  }
}

void transpose_kernelLauncher(
    c10::Half* input, // [batch_size x None]
    c10::Half* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int* size_dim2,
    const int* size_dim3,
    const cudaStream_t stream)
{
  dim3 grid;
  // Actually is batch size.
  grid.x = block_numel,

  transpose<32><<<grid, dim3(8, 32), 0, stream>>>(
      input,
      output,
      block_offsets,
      offsets,
      batch_size,
      size_dim2,
      size_dim3);
}

}
} // namespace nested_tensor
