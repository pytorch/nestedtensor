#pragma once
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

namespace nested_tensor {
namespace cuda {

void transpose_kernelLauncher(
    c10::Half* input,
    c10::Half* output,
    const int* lengths,
    const int batch_size,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream);

}
} // namespace nested_tensor
