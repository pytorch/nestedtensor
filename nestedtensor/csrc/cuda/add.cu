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
    const c10::Half* input,
    const c10::Half* mean,
    const c10::Half* running_var,
    const c10::Half eps,
    const c10::Half* weight,
    const c10::Half* bias,
    c10::Half* output,
    const int input_outer_stride,
    const int* offsets)
{
  const int batch_id  = blockIdx.x;
  const int grid_id  = blockIdx.y;
  const int scalars_id  = batch_id / input_outer_stride;
  const int grain_size = 256 * 2;
  const int tid = threadIdx.x + grid_id * 256;
  const int range = (offsets[batch_id + 1] - offsets[batch_id]);
  const int num_chunks = range / grain_size;
  c10::Half value = running_var[scalars_id] + eps;
  value = hrsqrt(value);
  value = value * weight[scalars_id];
  c10::Half value2 = mean[scalars_id] * value - bias[scalars_id];

  int input_offset = offsets[batch_id] + tid;
  int id = 0;
  for (; id < num_chunks; id++) {
    output[input_offset] = __ldg(reinterpret_cast<const __half*>(input) + input_offset) * value - value2;
    input_offset += grain_size;
  }
  if (input_offset < offsets[batch_id + 1]) { //leftover + tid < range) {
    output[input_offset] = __ldg(reinterpret_cast<const __half*>(input) + input_offset) * value - value2;
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
  grid.y = 2;

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
