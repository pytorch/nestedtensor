
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
    const int batch_size,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream);
}
}
