#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/cuda/attention.h>
#include <nestedtensor/csrc/cuda/cuda_kernels.h>
#include <nestedtensor/csrc/masking.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_functions.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>
#include <chrono>
namespace py = pybind11;

using namespace torch::nested_tensor;
using namespace at;

namespace torch {
namespace nested_tensor {

at::Tensor bt_min_mha(
    int64_t num_heads,
    int64_t head_dim,
    double dropout_p,
    bool training,
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor attr_kernel,
    at::Tensor attr_bias,
    double scaling,
    at::Tensor out_proj_weight,
    at::Tensor out_proj_bias) {
  // TODO: Assert that max seq_len is 1024!
  TORCH_CHECK(get_dim(query) == 3, "query needs to be 3 dim.");
  TORCH_CHECK(get_dim(key) == 3, "key needs to be 3 dim.");
  TORCH_CHECK(get_dim(value) == 3, "value needs to be 3 dim.");
  // TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  // auto opt_sizes = get_opt_sizes(query);
  // if (!opt_sizes[2]) {
  //   throw std::runtime_error("query's third dimension must be regular.");
  // }
  // TODO: Add explicit check that verifies query, key and value are the same
  // auto start = std::chrono::system_clock::now();
  auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  at::Tensor input_mask = to_mask(query, 2);
  input_mask = input_mask.to(options);
  int64_t batch_size = input_mask.size(0);
  int64_t seq_len = input_mask.size(1);
  int64_t embedding_dim = head_dim * num_heads; //*(opt_sizes[2]);
  int64_t head_num = num_heads;
  int64_t size_per_head = embedding_dim / head_num;
  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);

  int64_t input_tensor_size = batch_size * head_num * seq_len * size_per_head;
  int64_t attn_tensor_size = batch_size * head_num * seq_len * seq_len;
  int word_num = batch_size * seq_len;
  Tensor prefix_sum = torch::zeros({word_num}, options);
  Tensor batch_idx = torch::zeros({word_num}, options);
  Tensor word_idx = torch::zeros({word_num}, options);

  int* prefix_sum_ptr = prefix_sum.data_ptr<int>();
  int* batch_idx_ptr = batch_idx.data_ptr<int>();
  int* word_idx_ptr = word_idx.data_ptr<int>();

  at::Tensor tmp = get_buffer(query);

  auto query_esize = get_efficient_nested_size(query);
  TORCH_CHECK(query_esize.height() == 1, "Query nested dim isn't 1.");
  auto query_esize_sizes = query_esize.sizes();

  at::Tensor attr_mask = input_mask.view({-1, 1, 1, seq_len}).to(float_options);
  attr_mask = attr_mask * attr_mask.transpose(2, 3);

  nteffectivetransformer::exclusiveScan_kernelLauncher(
      prefix_sum_ptr,
      input_mask.data_ptr<int>(),
      input_mask.size(0) * input_mask.size(1),
      defaultStream);


  nteffectivetransformer::compressBertInput_kernelLauncher(
      input_mask.data_ptr<int>(),
      prefix_sum_ptr,
      batch_idx_ptr,
      word_idx_ptr,
      (int32_t)(batch_size),
      (int32_t)(seq_len),
      (int32_t)(embedding_dim),
      defaultStream);

  at::Tensor packed = at::matmul(query, attr_kernel.t());
  at::Tensor packed_buf = get_buffer(packed).contiguous().reshape({-1, 3 * embedding_dim});
  std::vector<at::Tensor> packed_chunks = packed_buf.chunk(3, -1);
  at::Tensor q_buf_ = packed_chunks[0].contiguous().reshape({-1});
  at::Tensor k_buf_ = packed_chunks[1].contiguous().reshape({-1});
  at::Tensor v_buf_ = packed_chunks[2].contiguous().reshape({-1});
  at::Tensor q = wrap_buffer(std::move(q_buf_), get_efficient_nested_size(query),
      get_efficient_nested_stride(query));
  at::Tensor k = wrap_buffer(std::move(k_buf_), get_efficient_nested_size(query),
      get_efficient_nested_stride(query));
  at::Tensor v = wrap_buffer(std::move(v_buf_), get_efficient_nested_size(query),
      get_efficient_nested_stride(query));

  std::vector<at::Tensor> bias_chunks = attr_bias.chunk(3);
  at::Tensor attr_bias_Q = bias_chunks[0];
  at::Tensor attr_bias_K = bias_chunks[1];
  at::Tensor attr_bias_V = bias_chunks[2];
  
  q = q + attr_bias_Q;
  k = k + attr_bias_K;
  v = v + attr_bias_V;

  at::Tensor query_buf = to_padded_tensor(q, 0).contiguous();
  at::Tensor key_buf = to_padded_tensor(k, 0).contiguous();
  at::Tensor val_buf = to_padded_tensor(v, 0).contiguous();
  query_buf = query_buf.reshape({batch_size, head_num, seq_len, size_per_head});
  key_buf = key_buf.reshape({batch_size, head_num, seq_len, size_per_head});
  val_buf = val_buf.reshape({batch_size, head_num, seq_len, size_per_head});
  query_buf = query_buf.to(float_options);
  key_buf = key_buf.to(float_options);
  val_buf = val_buf.to(float_options);

  int valid_word_num = get_numel(query) / embedding_dim;

  key_buf = key_buf.transpose(2, 3);
  at::Tensor attn_output_weights = at::matmul(query_buf, key_buf).contiguous();

  nteffectivetransformer::cuda::softmax_kernel_kernelLauncher<float>(
      attn_output_weights.data_ptr<float>(),
      attr_mask.data_ptr<float>(),
      batch_size,
      head_num,
      seq_len,
      (float)(scaling),
      defaultStream);

  auto attn_output = at::matmul(attn_output_weights, val_buf);
  at::Tensor attr_out = torch::zeros({valid_word_num, embedding_dim}, float_options);

  nteffectivetransformer::cuda::transpose_rm_padding_kernelLauncher<float>(
      attn_output.data_ptr<float>(),
      attr_out.data_ptr<float>(),
      valid_word_num,
      batch_size,
      seq_len,
      head_num,
      size_per_head,
      batch_idx_ptr,
      word_idx_ptr,
      defaultStream);

  // TODO: Bias is variably sized, need to add support for that.
  at::Tensor result = at::matmul(attr_out, out_proj_weight.t());
  result = result.reshape({-1});
  return wrap_buffer(
      std::move(result),
      get_efficient_nested_size(query),
      get_efficient_nested_stride(query));
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def(
      "bt_min_mha(int num_heads, int head_dim, float dropout_p, bool training, Tensor query, Tensor key, Tensor value, Tensor attr_kernel, Tensor attr_bias, float scaling, Tensor out_proj_weight, Tensor out_proj_bias) -> Tensor");
  m.impl("bt_min_mha", NestedTensorKey, &bt_min_mha);
}

} // namespace nested_tensor
} // namespace torch
