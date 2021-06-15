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

}
}
