/*
 * Copyright (C) 2020 ByteDance Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <nestedtensor/csrc/cuda/attention.h>
#include <nestedtensor/csrc/cuda/bert_transformer_op.h>
#include <nestedtensor/csrc/cuda/common.h>
#include <nestedtensor/csrc/cuda/cuda_kernels.h>
#include <string>
#include <type_traits>
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
    int* batch_idx,
    int* word_idx,
    DataType_* attr_mask,
    int64_t batch_size_,
    int64_t head_num_,
    int64_t seq_len_,
    int64_t size_per_head_,
    int64_t valid_word_num_) {
  at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(stream);
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

  check_cuda_error(cublasSetStream(cublas_handle, stream));

  /// 1. Set compute type
  cudaDataType_t computeType, AType, BType, CType;
  int cublasAlgo[3];
  if constexpr (std::is_same<DataType_, float>::value) {
    computeType = CUDA_R_32F;
    AType = CUDA_R_32F;
    BType = CUDA_R_32F;
    CType = CUDA_R_32F;
    cublasAlgo[0] = -1;
    cublasAlgo[1] = -1;
    cublasAlgo[2] = -1;
  } else {
    computeType = CUDA_R_16F;
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
    cublasAlgo[0] = 99;
    cublasAlgo[1] = 99;
    cublasAlgo[2] = 99;
  }
  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  /// 2. allocate buffer for transformer
  Tensor buf_tensor;
  int batch_size = batch_size_;
  int head_num = head_num_;
  int from_seq_len = seq_len_;
  int size_per_head = size_per_head_;
  int input_tensor_size = batch_size * head_num * from_seq_len * size_per_head;
  int attn_tensor_size = batch_size * head_num * from_seq_len * from_seq_len;
  long long int buf_size = input_tensor_size * 13 + attn_tensor_size;
  Tensor buf_tensor = torch::empty({buf_size});

  /// 3. assign intermediate pointer
  DataType_* buf = buf_tensor.data_ptr<DataType_>();
  /// buffer for qkv
  DataType_* query_buf_ = buf + 0 * input_tensor_size;
  DataType_* key_buf_ = buf + 1 * input_tensor_size;
  DataType_* value_buf_ = buf + 2 * input_tensor_size;
  DataType_* query_ = buf + 3 * input_tensor_size;
  DataType_* key_ = buf + 4 * input_tensor_size;
  DataType_* value_ = buf + 5 * input_tensor_size;
  /// buffer for self attention
  DataType_* qk_buf_ = buf + 0 * input_tensor_size;
  DataType_* transpose_dst_ =
      buf + std::max(attn_tensor_size, input_tensor_size);
  /// buffer for output matmat
  DataType_* attr_out_buf_ = buf + 0 * input_tensor_size;
  DataType_* attr_matmul_buf_ = buf + 1 * input_tensor_size;
  DataType_* inter_matmul_buf_ = buf + 2 * input_tensor_size;

  /// 4. get valid word num
  int valid_word_num = valid_word_num_;

  // 5. input -> Q K V
  {
    int m = valid_word_num;
    int k = head_num * size_per_head;
    int n = k;

    check_cuda_error(cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        attr_kernel_Q,
        AType,
        n,
        from_tensor,
        BType,
        k,
        &beta,
        query_buf_,
        CType,
        n,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

    check_cuda_error(cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        attr_kernel_K,
        AType,
        n,
        to_tensor,
        BType,
        k,
        &beta,
        key_buf_,
        CType,
        n,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

    check_cuda_error(cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        attr_kernel_V,
        AType,
        n,
        to_tensor,
        BType,
        k,
        &beta,
        value_buf_,
        CType,
        n,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

    // check_cuda_error(cudaMemsetAsync(query_, 0, input_tensor_size *
    // sizeof(DataType_), stream)); check_cuda_error(cudaMemsetAsync(key_, 0,
    // input_tensor_size * sizeof(DataType_), stream));
    // check_cuda_error(cudaMemsetAsync(value_, 0, input_tensor_size *
    // sizeof(DataType_), stream));
    check_cuda_error(cudaMemsetAsync(
        query_, 0, 3 * input_tensor_size * sizeof(DataType_), stream));

    /// add bias & add padding & transpose for self-attention
    cuda::add_QKV_bias_padding_kernelLauncher<DataType_>(
        query_buf_,
        attr_bias_Q,
        key_buf_,
        attr_bias_K,
        value_buf_,
        attr_bias_V,
        query_,
        key_,
        value_,
        valid_word_num,
        batch_size,
        from_seq_len,
        head_num,
        size_per_head,
        batch_idx,
        word_idx,
        stream);
  }

  /// 6. self-attention
  {
    check_cuda_error(cublasGemmStridedBatchedEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        from_seq_len,
        from_seq_len,
        size_per_head,
        &alpha,
        key_,
        AType,
        size_per_head,
        from_seq_len * size_per_head,
        query_,
        BType,
        size_per_head,
        from_seq_len * size_per_head,
        &beta,
        qk_buf_,
        CType,
        from_seq_len,
        from_seq_len * from_seq_len,
        batch_size * head_num,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo[1])));

    DataType_ scaler = 1 / sqrtf(size_per_head * 1.0f);
    cuda::softmax_kernel_kernelLauncher<DataType_>(
        qk_buf_,
        attr_mask,
        batch_size,
        head_num,
        from_seq_len,
        scaler,
        stream);

    check_cuda_error(cublasGemmStridedBatchedEx(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        size_per_head,
        from_seq_len,
        from_seq_len,
        &alpha,
        value_,
        AType,
        size_per_head,
        from_seq_len * size_per_head,
        qk_buf_,
        BType,
        from_seq_len,
        from_seq_len * from_seq_len,
        &beta,
        transpose_dst_,
        CType,
        size_per_head,
        from_seq_len * size_per_head,
        batch_size * head_num,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo[2])));

    cuda::transpose_rm_padding_kernelLauncher<DataType_>(
        transpose_dst_,
        attr_out_buf_,
        valid_word_num,
        batch_size,
        from_seq_len,
        head_num,
        size_per_head,
        batch_idx,
        word_idx,
        stream);
  }

  //  /// 7. matmat & layer norm
  //  {
  //    int m = valid_word_num;
  //    int k = head_num * size_per_head;
  //    int n = k;
  //
  //    check_cuda_error(cublasGemmEx(
  //        param.cublas_handle,
  //        CUBLAS_OP_N,
  //        CUBLAS_OP_N,
  //        n,
  //        m,
  //        k,
  //        &alpha,
  //        param.attr_output_kernel,
  //        AType,
  //        n,
  //        attr_out_buf_,
  //        BType,
  //        k,
  //        &beta,
  //        attr_matmul_buf_,
  //        CType,
  //        n,
  //        computeType,
  //        static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));
  //
  //    add_bias_input_layernorm_kernelLauncher<DataType_>(
  //        attr_matmul_buf_,
  //        param.from_tensor,
  //        param.attr_output_bias,
  //        param.attr_output_layernorm_gamma,
  //        param.attr_output_layernorm_beta,
  //        m,
  //        n,
  //        param.stream);
  //
  //    n *= 4;
  //    check_cuda_error(cublasGemmEx(
  //        param.cublas_handle,
  //        CUBLAS_OP_N,
  //        CUBLAS_OP_N,
  //        n,
  //        m,
  //        k,
  //        &alpha,
  //        param.inter_kernel,
  //        AType,
  //        n,
  //        attr_matmul_buf_,
  //        BType,
  //        k,
  //        &beta,
  //        inter_matmul_buf_,
  //        CType,
  //        n,
  //        computeType,
  //        static_cast<cublasGemmAlgo_t>(cublasAlgo[1])));
  //
  //    add_bias_act_kernelLauncher<DataType_>(
  //        inter_matmul_buf_, param.inter_bias, m, n, param.stream);
  //
  //    n = k;
  //    k *= 4;
  //    check_cuda_error(cublasGemmEx(
  //        param.cublas_handle,
  //        CUBLAS_OP_N,
  //        CUBLAS_OP_N,
  //        n,
  //        m,
  //        k,
  //        &alpha,
  //        param.output_kernel,
  //        AType,
  //        n,
  //        inter_matmul_buf_,
  //        BType,
  //        k,
  //        &beta,
  //        param.transformer_out,
  //        CType,
  //        n,
  //        computeType,
  //        static_cast<cublasGemmAlgo_t>(cublasAlgo[2])));
  //
  //    add_bias_input_layernorm_kernelLauncher<DataType_>(
  //        param.transformer_out,
  //        attr_matmul_buf_,
  //        param.output_bias,
  //        param.output_layernorm_gamma,
  //        param.output_layernorm_beta,
  //        m,
  //        n,
  //        param.stream);
  //  }
};
} // namespace effectivetransformer
