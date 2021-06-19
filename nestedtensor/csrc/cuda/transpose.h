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
    const int* offsets,
    const int batch_size,
    const int num_channel,
    const cudaStream_t stream);

}
} // namespace nested_tensor