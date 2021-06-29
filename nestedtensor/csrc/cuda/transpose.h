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
    const int* block_offsets,
    const int* offsets,
    // const int* blocks2,
    // const int* blocks3,
    const int* blocks_batch_dim,
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int block_numel,
    const int numel,
    const cudaStream_t stream);

}
} // namespace nested_tensor
