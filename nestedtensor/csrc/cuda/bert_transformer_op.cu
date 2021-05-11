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

#include <nestedtensor/csrc/cuda/bert_transformer_op.h>
namespace effectivetransformer {

template <typename DataType_>
at::Tensor bt_mha(
    DataType_* from_tensor,
    DataType_* query_buf_,
    DataType_* key_buf_,
    DataType_* value_buf_,
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
    DataType_* buf,
    DataType_ scaler,
    int* prefix_sum_ptr,
    int* input_mask_ptr,
    int word_num) {
  at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(stream);
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  check_cuda_error(cublasSetStream(cublas_handle, stream));

  /// 4. get valid word num
  // int valid_word_num = valid_word_num_;
  int valid_word_num;
  check_cuda_error(cudaMemcpyAsync(
    &valid_word_num,
    prefix_sum_ptr + word_num - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
  int last_mask;
  check_cuda_error(cudaMemcpyAsync(
    &last_mask,
    input_mask_ptr + word_num - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
  if (last_mask == 1) {
    valid_word_num++;
  }


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
  int batch_size = batch_size_;
  int head_num = head_num_;
  int from_seq_len = seq_len_;
  int size_per_head = size_per_head_;
  int input_tensor_size = batch_size * head_num * from_seq_len * size_per_head;
  int attn_tensor_size = batch_size * head_num * from_seq_len * from_seq_len;

   DataType_* query_         = buf + 0 * input_tensor_size;
   DataType_* key_           = buf + 1 * input_tensor_size;
   DataType_* value_         = buf + 2 * input_tensor_size;
   /// buffer for self attention
   DataType_* qk_buf_           = buf + 3 * input_tensor_size;
   /// buffer for output matmat
   DataType_* attr_out_buf_     = buf + 4 * input_tensor_size;
   DataType_* transpose_dst_    = buf + 5 * input_tensor_size;

  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  int64_t result_numel = valid_word_num * head_num_ * size_per_head_;
  at::Tensor result = torch::empty({result_numel}, float_options);
  DataType_* attr_matmul_buf_ = result.data_ptr<float>();

  // 5. input -> Q K V
  {
    int m = valid_word_num;
    int k = head_num * size_per_head;
    int n = k;

  // std::cout << "014" << std::endl;
  // stream.synchronize();

    // check_cuda_error(cudaMemsetAsync(query_, 0, input_tensor_size * sizeof(DataType_), stream));
    // check_cuda_error(cudaMemsetAsync(key_, 0, input_tensor_size * sizeof(DataType_), stream));
    // check_cuda_error(cudaMemsetAsync(value_, 0, input_tensor_size * sizeof(DataType_), stream));
    check_cuda_error(cudaMemsetAsync(
        query_, 0, 3 * input_tensor_size * sizeof(DataType_), stream));
  // std::cout << "015" << std::endl;
  // stream.synchronize();

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
  // std::cout << "016" << std::endl;
  // stream.synchronize();
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
  // std::cout << "017" << std::endl;
  // stream.synchronize();

    // DataType_ scaler = 1 / sqrtf(size_per_head * 1.0f);
    // DataType_ scaler = 1;
    // DataType_ scaler = 1 / sqrtf(size_per_head * 1.0f);
  // std::cout << "018" << std::endl;
  // stream.synchronize();
     cuda::softmax_kernel_kernelLauncher<DataType_>(
         qk_buf_, attr_mask, batch_size, head_num, from_seq_len, scaler, stream);
  // std::cout << "019" << std::endl;
  // stream.synchronize();

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
  // std::cout << "020" << std::endl;
  // stream.synchronize();

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
  // std::cout << "021" << std::endl;
  // stream.synchronize();
  }

   /// 7. matmat & layer norm
   {
     int m = valid_word_num;
     int k = head_num * size_per_head;
     int n = k;
     // TODO: Currently does not support bias!
  
     check_cuda_error(cublasGemmEx(
         cublas_handle,
         CUBLAS_OP_N,
         CUBLAS_OP_N,
         n,
         m,
         k,
         &alpha,
         attr_output_kernel,
         AType,
         n,
         attr_out_buf_,
         BType,
         k,
         &beta,
         attr_matmul_buf_,
         CType,
         n,
         computeType,
         static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));
  // stream.synchronize();
   }
   return result;
};

template at::Tensor bt_mha<float>(
    float* from_tensor,
    float* query_buf_,
    float* key_buf_,
    float* value_buf_,
    float* to_tensor,
    float* attr_bias_Q,
    float* attr_bias_K,
    float* attr_bias_V,
    float* attr_output_kernel,
    int* batch_idx,
    int* word_idx,
    float* attr_mask,
    int64_t batch_size_,
    int64_t head_num_,
    int64_t seq_len_,
    int64_t size_per_head_,
    float* buf,
    float scaler,
    int* prefix_sum_ptr,
    int* input_mask_ptr,
    int word_num);
} // namespace effectivetransformer
