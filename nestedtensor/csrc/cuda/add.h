#pragma once
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

namespace nested_tensor {
namespace cuda {

void add_scalar_kernelLauncher(
    c10::Half* input,
    c10::Half* scalars,
    c10::Half* output,
    const int batch_size,
    const int input_outer_stride,
    const int* offsets,
    const cudaStream_t stream);

void mul_scalar_kernelLauncher(
    c10::Half* input,
    c10::Half* scalars,
    c10::Half* output,
    const int batch_size,
    const int input_outer_stride,
    const int* offsets,
    const cudaStream_t stream);

void sub_scalar_kernelLauncher(
    c10::Half* input,
    c10::Half* scalars,
    c10::Half* output,
    const int batch_size,
    const int input_outer_stride,
    const int* offsets,
    const cudaStream_t stream);

void batchnorm_inference_kernelLauncher(
    c10::Half* input,
    c10::Half* mean,
    // c10::Half* invstd,
    c10::Half* running_var,
    c10::Half eps,
    c10::Half* weight,
    c10::Half* bias,
    c10::Half* output,
    const int batch_size,
    const int num_scalars,
    const int numel,
    const int* offsets,
    const cudaStream_t stream);

void batchnorm_inference_channels_last_kernelLauncher(
    c10::Half* input,
    c10::Half* mean,
    c10::Half* running_var,
    c10::Half eps,
    c10::Half* weight,
    c10::Half* bias,
    c10::Half* output,
    const int num_channel,
    const int numel,
    const cudaStream_t stream);

}
}
