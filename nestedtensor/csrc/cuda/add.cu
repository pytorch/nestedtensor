#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/attention.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

__global__
void add_scalars(
    const __half* input,
    const __half* scalars,
    __half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int scalars_id  = batch_id / input_outer_stride;
  // const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  // const int num_chunks = range / grain_size;
  // printf("scalars_id: %d\n", scalars_id);
  // // printf("batch_id: %d , input_outer_stride: %d , scalars_id: %d\n",
  // //     batch_id, input_outer_stride, scalars_id);
  for (int id = offsets[batch_id]; id < offsets[batch_id + 1]; id++) {
    // printf("id: %d , tid: %d\n", id, tid);
    output[id] = __hadd(input[id], scalars[scalars_id]);
  }
  // const int leftover = num_chunks * grain_size;
  // if (leftover + tid < range) {
  //   output[offsets[batch_id] + leftover + tid]
  //     = input[offsets[batch_id] + leftover + tid];
  //     // = __hadd(input[offsets[batch_id] + leftover + tid], scalars[scalars_id]);
  // }
}

void add_scalar_kernelLauncher(
    __half* input, // [batch_size x offsets[-1]]
    __half* scalars, // [batch_size]
    __half* output, // [batch_size x offsets[-1]]
    const int batch_size,
    const int input_outer_stride,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;
  // printf("batch_size: %d\n", batch_size);

  add_scalars<<<grid, 1, 0, stream>>>(
      input,
      scalars,
      output,
      input_outer_stride,
      offsets);
}

__global__
void mul_scalars(
    const __half* input,
    const __half* scalars,
    __half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int scalars_id  = batch_id / input_outer_stride;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  for (int id = offsets[batch_id]; id < offsets[batch_id + 1]; id++) {
    output[id] = __hmul(input[id], scalars[scalars_id]);
  }
}

void mul_scalar_kernelLauncher(
    __half* input, // [batch_size x offsets[-1]]
    __half* scalars, // [batch_size]
    __half* output, // [batch_size x offsets[-1]]
    const int batch_size,
    const int input_outer_stride,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  mul_scalars<<<grid, 1, 0, stream>>>(
      input,
      scalars,
      output,
      input_outer_stride,
      offsets);
}

}
}
