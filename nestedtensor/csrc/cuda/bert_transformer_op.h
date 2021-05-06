#pragma once
#include <cublas_v2.h>
namespace effectivetransformer {
template <typename DataType_>
void bt_mha(
    DataType_* from_tensor,
    DataType_* attr_kernel_Q,
    DataType_* attr_kernel_K,
    DataType_* attr_kernel_V,
    DataType_* to_tensor,
    DataType_* attr_bias_Q,
    DataType_* attr_bias_K,
    DataType_* attr_bias_V,
    DataType_* attr_output_kernel,
    int* batch_idx,
    int* word_idx,
    DataType_* attr_mask,
    int64_t batch_size_,
    int64_t head_num_,
    int64_t seq_len_,
    int64_t size_per_head_,
    int64_t valid_word_num_,
    DataType_* buf,
    DataType_ scaler);
}
