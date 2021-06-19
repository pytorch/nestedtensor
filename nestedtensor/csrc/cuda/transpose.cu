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
    const int batch_size,
    const int num_channel)
{
  const int batch_id  = blockIdx.x;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  const int num_chunks = range / grain_size;
  // for (int id = 0; id < num_chunks; id++) {
  //   output[batch_id * output_stride + id * grain_size + tid]
  //     = input[offsets[batch_id] * inner_size + id * grain_size + tid];
  // }
  // const int leftover = num_chunks * grain_size;
  // if (leftover + tid < range) {
  //   output[batch_id * output_stride + leftover + tid]
  //     = input[offsets[batch_id] * inner_size + leftover + tid];
  // }
}

void transpose_kernelLauncher(
    c10::Half* input, // [batch_size x None]
    c10::Half* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets, // [batch_size]
    const int batch_size,
    const int num_channel,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  transpose<<<grid, 1, 0, stream>>>(
      input,
      output,
      offsets,
      batch_size,
      num_channel);
}

}
} // namespace nested_tensor
