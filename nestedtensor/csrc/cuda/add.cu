#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/add.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

__global__
void add_scalars(
    c10::Half* input,
    c10::Half* scalars,
    c10::Half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int scalars_id  = batch_id / input_outer_stride;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  const int num_chunks = range / grain_size;
  for (int id = 0; id < num_chunks; id++) {
    output[offsets[batch_id] + id * grain_size + tid] =
      input[offsets[batch_id] + id * grain_size + tid] + scalars[scalars_id];
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < range) {
    output[offsets[batch_id] + leftover + tid] =
      input[offsets[batch_id] + leftover + tid] + scalars[scalars_id];
  }
}

void add_scalar_kernelLauncher(
    c10::Half* input, // [batch_size x offsets[-1]]
    c10::Half* scalars, // [batch_size]
    c10::Half* output, // [batch_size x offsets[-1]]
    const int batch_size,
    const int input_outer_stride,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  add_scalars<<<grid, 256, 0, stream>>>(
      input,
      scalars,
      output,
      input_outer_stride,
      offsets);
}

__global__
void mul_scalars(
    c10::Half* input,
    c10::Half* scalars,
    c10::Half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int scalars_id  = batch_id / input_outer_stride;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  const int num_chunks = range / grain_size;
  for (int id = 0; id < num_chunks; id++) {
    output[offsets[batch_id] + id * grain_size + tid] =
      input[offsets[batch_id] + id * grain_size + tid] * scalars[scalars_id];
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < range) {
    output[offsets[batch_id] + leftover + tid] =
      input[offsets[batch_id] + leftover + tid] * scalars[scalars_id];
  }
}

void mul_scalar_kernelLauncher(
    c10::Half* input, // [batch_size x offsets[-1]]
    c10::Half* scalars, // [batch_size]
    c10::Half* output, // [batch_size x offsets[-1]]
    const int batch_size,
    const int input_outer_stride,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  mul_scalars<<<grid, 256, 0, stream>>>(
      input,
      scalars,
      output,
      input_outer_stride,
      offsets);
}

__global__
void sub_scalars(
    c10::Half* input,
    c10::Half* scalars,
    c10::Half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int scalars_id  = batch_id / input_outer_stride;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  const int num_chunks = range / grain_size;
  for (int id = 0; id < num_chunks; id++) {
    output[offsets[batch_id] + id * grain_size + tid] =
      input[offsets[batch_id] + id * grain_size + tid] - scalars[scalars_id];
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < range) {
    output[offsets[batch_id] + leftover + tid] =
      input[offsets[batch_id] + leftover + tid] - scalars[scalars_id];
  }
}

void sub_scalar_kernelLauncher(
    c10::Half* input, // [batch_size x offsets[-1]]
    c10::Half* scalars, // [batch_size]
    c10::Half* output, // [batch_size x offsets[-1]]
    const int batch_size,
    const int input_outer_stride,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  sub_scalars<<<grid, 256, 0, stream>>>(
      input,
      scalars,
      output,
      input_outer_stride,
      offsets);
}

__global__
void batchnorm_inference(
    c10::Half* input,
    c10::Half* mean,
    c10::Half* running_var,
    c10::Half eps,
    c10::Half* weight,
    c10::Half* bias,
    c10::Half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int scalars_id  = batch_id / input_outer_stride;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  const int num_chunks = range / grain_size;
  c10::Half value = running_var[scalars_id] + eps;
  value = hrsqrt(value);
  value = value * weight[scalars_id];
  for (int id = 0; id < num_chunks; id++) {
    output[offsets[batch_id] + id * grain_size + tid] =
      (((input[offsets[batch_id] + id * grain_size + tid] - mean[scalars_id])
       * value)
       + bias[scalars_id]);
  }
  const int leftover = num_chunks * grain_size;
  if (leftover + tid < range) {
    output[offsets[batch_id] + leftover + tid] =
      (((input[offsets[batch_id] + leftover + tid] - mean[scalars_id])
       * value)
       + bias[scalars_id]);
  }
}

void batchnorm_inference_kernelLauncher(
    c10::Half* input, // [batch_size x offsets[-1]]
    c10::Half* mean, // [batch_size]
    c10::Half* running_var,
    c10::Half eps,
    c10::Half* weight, // [batch_size]
    c10::Half* bias, // [batch_size]
    c10::Half* output, // [batch_size x offsets[-1]]
    const int batch_size,
    const int input_outer_stride,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  batchnorm_inference<<<grid, 256, 0, stream>>>(
      input,
      mean,
      running_var,
      eps,
      weight,
      bias,
      output,
      input_outer_stride,
      offsets);
}

}
}
