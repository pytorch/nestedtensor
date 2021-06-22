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
  // const int grain_size = blockDim.x;
  // const int tid = threadIdx.x;
  // const int range = (offsets[(batch_id + 1) * num_channel] - offsets[batch_id * num_channel]);
  // for (int channel_id = 0; channel_id < num_channel; channel_id++) {
    //printf("batch_id: %d, channel_id: %d, offsets[%d]: %d, offsets[%d]: %d, sizes_dim2[%d]: %d, sizes_dim3[%d]: %d, strides_dim2[%d]: %d, strides_dim3[%d]: %d\n",
    //    batch_id, channel_id, batch_id, offsets[batch_id],
    //    (batch_id + 1), offsets[(batch_id + 1)],
    //    batch_id, sizes_dim2[batch_id], batch_id, sizes_dim3[batch_id],
    //    batch_id, strides_dim2[batch_id], batch_id, strides_dim3[batch_id]);
    // int64_t size2 = sizes_dim2[batch_id];
    int size2 = sizes_dim2[batch_id];
    int size3 = sizes_dim3[batch_id];
    for (int ii = 0; ii < size2 * size3; ii++) {
      int j = (ii % size2) * size3;
      int i = (ii / size2);


      printf("size2: %d size3: %i stride0: %i stride1: %i \n",
       size2,
       size3,
       offsets[batch_id] + j + i,
       offsets[batch_id] + ii
          );
       output[offsets[batch_id] + j + i] = 
        input[offsets[batch_id] + ii];
    }
  // }
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
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  transpose<<<grid, 1, 0, stream>>>(
      input,
      output,
      offsets,
      sizes_dim2,
      sizes_dim3,
      batch_size);
}

}
} // namespace nested_tensor
