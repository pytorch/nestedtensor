#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>
#include <vector>

namespace nested_tensor {
namespace cuda {

template <typename T>
void add_padding_kernelLauncher(
    T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    std::vector<int64_t> output_sizes,
    const int batch_size,
    const cudaStream_t stream);

template <typename T>
void add_padding_mask_kernelLauncher(
    T* input,
    T* output,
    int* output_mask,
    const int* lengths,
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream);

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream);

}
} // namespace nested_tensor
