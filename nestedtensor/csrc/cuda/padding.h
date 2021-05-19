#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace nested_tensor {
namespace cuda {

template <typename T>
void add_padding_kernelLauncher(
    T* input,
    T* output,
    const int* lengths,
    int batch_size,
    int output_stride,
    const cudaStream_t stream);
}
} // namespace nested_tensor
