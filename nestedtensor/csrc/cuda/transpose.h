#pragma once
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

namespace nested_tensor {
namespace cuda {

template <typename T>
void transpose_kernelLauncher(
    T* input,
    T* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int* sizes_dim2,
    const int* sizes_dim3,
    const cudaStream_t stream);

template <typename T>
void transpose_nchw_nhwc_kernelLauncher(
    T* input,
    T* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int sizes_dim2,
    const int* sizes_dim3,
    const cudaStream_t stream);

}
} // namespace nested_tensor
