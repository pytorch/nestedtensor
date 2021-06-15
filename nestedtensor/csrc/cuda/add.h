#pragma once
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

namespace nested_tensor {
namespace cuda {

void add_scalar_kernelLauncher(
    __half* input,
    __half* scalars,
    __half* output,
    const int batch_size,
    const int input_outer_stride,
    const int* offsets,
    const cudaStream_t stream);

void mul_scalar_kernelLauncher(
    __half* input,
    __half* scalars,
    __half* output,
    const int batch_size,
    const int input_outer_stride,
    const int* offsets,
    const cudaStream_t stream);

}
}
