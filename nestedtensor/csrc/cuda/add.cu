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

template <int num_threads>
__global__
void batchnorm_inference(
    const c10::Half* input,
    const c10::Half* mean,
    const c10::Half* running_var,
    const c10::Half eps,
    const c10::Half* weight,
    const c10::Half* bias,
    c10::Half* output,
    const int num_scalars,
    const int* offsets)
{
  const int batch_id = blockIdx.x;
  const int scalars_id  = blockIdx.y;
  const int grain_size = num_threads;
  const int tid = threadIdx.x;
  const int offset_id = batch_id * num_scalars + scalars_id;
  const int range = (offsets[offset_id + 1] - offsets[offset_id]);
  const int num_chunks = range / grain_size;
  c10::Half value = running_var[scalars_id] + eps;
  value = __frsqrt_rn(value);
  value = value * weight[scalars_id];
  c10::Half value2 = mean[scalars_id] * value - bias[scalars_id];

  int input_offset = offsets[offset_id] + tid;
  int id = 0;
  for (; id < num_chunks; id++) {
    output[input_offset] = input[input_offset] * value - value2;
    input_offset += grain_size;
  }
  if (input_offset < offsets[offset_id + 1]) {
    output[input_offset] = input[input_offset] * value - value2;
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
    const int num_scalars,
    const int numel,
    const int* offsets /* [batch_size] */,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;
  grid.y = num_scalars;

  batchnorm_inference<32><<<grid, 32, 0, stream>>>(
      input,
      mean,
      running_var,
      eps,
      weight,
      bias,
      output,
      num_scalars,
      offsets);
}

template <int chunk_size, int num_threads>
__global__
void batchnorm_inference_channels_last(
    const c10::Half* input,
    const c10::Half* mean,
    const c10::Half* running_var,
    const c10::Half eps,
    const c10::Half* weight,
    const c10::Half* bias,
    c10::Half* output,
    const int num_channel,
    const int numel)
{
  const int block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int slice_offset = block_id * chunk_size;
  const int num_slices = numel / num_channel;
  if (slice_offset + chunk_size < num_slices) {
    for (int scalars_id = tid; scalars_id < num_channel; scalars_id += num_threads) {
      c10::Half value = running_var[scalars_id] + eps;
      value = __frsqrt_rn(value);
      value = value * weight[scalars_id];
      c10::Half value2 = mean[scalars_id] * value - bias[scalars_id];
      int offset = slice_offset * num_channel + scalars_id;
#pragma unroll
      for (int i = 0; i < chunk_size; i++) {
        output[offset] = input[offset] * value - value2;
        offset += num_channel;
      }
    }
  } else {
    for (int scalars_id = tid; scalars_id < num_channel; scalars_id += num_threads) {
      c10::Half value = running_var[scalars_id] + eps;
      value = __frsqrt_rn(value);
      value = value * weight[scalars_id];
      c10::Half value2 = mean[scalars_id] * value - bias[scalars_id];
#pragma unroll
      for (int i = 0; i < chunk_size; i++) {
        const int slice_id = slice_offset + i;
        if (slice_id < num_slices) {
          const int offset = slice_id * num_channel + scalars_id;
          output[offset] = input[offset] * value - value2;
        }
      }
    }
  }
}

void batchnorm_inference_channels_last_kernelLauncher(
    c10::Half* input, // [batch_size x offsets[-1]]
    c10::Half* mean, // [batch_size]
    c10::Half* running_var,
    c10::Half eps,
    c10::Half* weight, // [batch_size]
    c10::Half* bias, // [batch_size]
    c10::Half* output, // [batch_size x offsets[-1]]
    const int num_channel,
    const int numel,
    const cudaStream_t stream)
{
  dim3 grid;
  const int chunk_size = 32;
  const int slice_size = numel / num_channel;
  const int num_blocks = (slice_size + chunk_size - 1) / chunk_size;
  // At least 3 blocks per SM on Volta
  if (num_blocks < 240) {
    const int chunk_size = 16;
    const int slice_size = numel / num_channel;
    const int num_blocks = (slice_size + chunk_size - 1) / chunk_size;
    grid.x = num_blocks;
    batchnorm_inference_channels_last<16, 256><<<grid, 256, 0, stream>>>(
        input,
        mean,
        running_var,
        eps,
        weight,
        bias,
        output,
        num_channel,
        numel);
    return;
  }
  grid.x = num_blocks;

  batchnorm_inference_channels_last<32, 256><<<grid, 256, 0, stream>>>(
      input,
      mean,
      running_var,
      eps,
      weight,
      bias,
      output,
      num_channel,
      numel);
}

}
}
