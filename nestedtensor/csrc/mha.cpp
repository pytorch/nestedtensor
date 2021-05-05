#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/cuda/attention.h>
#include <nestedtensor/csrc/cuda/bert_transformer_op.h>
#include <nestedtensor/csrc/cuda/cuda_kernels.h>
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

at::Tensor min_mha(
    int64_t num_heads,
    int64_t head_dim,
    double dropout_p,
    bool training,
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor in_proj_weight,
    c10::optional<at::Tensor> in_proj_bias,
    double scaling,
    at::Tensor out_proj_weight,
    at::Tensor out_proj_bias) {
  TORCH_CHECK(query.dim() == 3, "query needs to be 3 dim.");
  TORCH_CHECK(key.dim() == 3, "key needs to be 3 dim.");
  TORCH_CHECK(value.dim() == 3, "value needs to be 3 dim.");
  TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  auto opt_sizes = get_opt_sizes(query);
  if (!opt_sizes[2]) {
    throw std::runtime_error("query's third dimension must be regular.");
  }
  int64_t edim = *(opt_sizes[2]);

  scaling = 1.0;
  at::Tensor q, k, v;
  q = at::addmm(
      at::slice(*in_proj_bias, 0, 0, edim),
      query,
      at::slice(in_proj_weight, 0, 0, edim).t(),
      scaling,
      scaling);
  k = at::addmm(
      at::slice(*in_proj_bias, 0, edim, 2 * edim),
      key,
      at::slice(in_proj_weight, 0, edim, 2 * edim).t());
  v = at::addmm(
      at::slice(*in_proj_bias, 0, 2 * edim),
      value,
      at::slice(in_proj_weight, 0, 2 * edim).t());

  q = q.reshape({-1, -1, num_heads, head_dim}).transpose(1, 2);
  k = k.reshape({-1, -1, num_heads, head_dim}).transpose(1, 2);
  v = v.reshape({-1, -1, num_heads, head_dim}).transpose(1, 2);
  auto attn_output_weights = at::matmul(q, k.transpose(2, 3));
  attn_output_weights = at::softmax(attn_output_weights, -1);
  attn_output_weights = at::dropout(attn_output_weights, dropout_p, training);
  auto attn_output = at::matmul(attn_output_weights, v);
  attn_output = attn_output.transpose(1, 2).reshape({-1, -1, edim});
  // std::cout << "out_proj_bias: " << out_proj_bias << std::endl;
  attn_output = at::addmm(out_proj_bias, attn_output, out_proj_weight.t());
  return attn_output;
}

// Mask is of shape (batch_size, seq_len) and type int32
// Returns prefix scan buffer of size (batch-size * seq_len * 2,)
Tensor exclusive_scan(Tensor mask) {
  Tensor prefix_sum_buf =
      torch::empty({mask.size(0) * mask.size(1) * 2}, mask.options());
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  effectivetransformer::exclusiveScan_kernelLauncher(
      prefix_sum_buf.data_ptr<int>(),
      mask.data_ptr<int>(),
      mask.size(0) * mask.size(1),
      defaultStream);
  return prefix_sum_buf;
}

std::tuple<Tensor, int64_t, int64_t> compress_bert_input(
    Tensor input, // float - (batch_size, seq_len, hidden_dim)
    Tensor mask, // int32 - (batch_size, seq_len)
    Tensor prefix_sum, // int32
    Tensor result, // float - (batch_size * num_head * seq_len * size_per_head)
    Tensor batch_idx, // int32 - (batch_size, seq_len)
    Tensor word_idx, // int32 - (batch_size, seq_len)
    int64_t batch_size,
    int64_t seq_len,
    int64_t hidden_dim) {
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  effectivetransformer::compressBertInput_kernelLauncher(
      input.data_ptr<float>(),
      mask.data_ptr<int>(),
      prefix_sum.data_ptr<int>(),
      result.data_ptr<float>(),
      batch_idx.data_ptr<int>(),
      word_idx.data_ptr<int>(),
      (int32_t)(batch_size),
      (int32_t)(seq_len),
      (int32_t)(hidden_dim),
      defaultStream);
  int word_num = batch_size * seq_len;
  int valid_word_num = prefix_sum.reshape({-1})[word_num - 1].item<int>();
  int last_mask = mask.reshape({-1})[word_num - 1].item<int>();
  if (last_mask == 1) {
    valid_word_num++;
  }
  return std::make_tuple(
      result, (int64_t)(valid_word_num), (int64_t)(last_mask));
}

Tensor restore_bert_output(
    Tensor result, // float - (batch_size * num_head * seq_len * size_per_head)
    Tensor input, // float - (batch_size, seq_len, hidden_dim)
    Tensor batch_idx, // int32 - (batch_size, seq_len)
    Tensor word_idx, // int32 - (batch_size, seq_len)
    int64_t valid_word_num,
    int64_t seq_len,
    int64_t hidden_dim) {
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  result.zero_();
  effectivetransformer::restoreBertOutput_kernelLauncher(
      result.data_ptr<float>(),
      input.data_ptr<float>(),
      batch_idx.data_ptr<int>(),
      word_idx.data_ptr<int>(),
      (int32_t)(valid_word_num),
      (int32_t)(seq_len),
      (int32_t)(hidden_dim),
      defaultStream);
  return result;
}

at::Tensor bt_min_mha(
    int64_t num_heads,
    int64_t head_dim,
    double dropout_p,
    bool training,
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor in_proj_weight,
    c10::optional<at::Tensor> in_proj_bias,
    double scaling,
    at::Tensor out_proj_weight,
    at::Tensor out_proj_bias) {
  TORCH_CHECK(query.dim() == 3, "query needs to be 3 dim.");
  TORCH_CHECK(key.dim() == 3, "key needs to be 3 dim.");
  TORCH_CHECK(value.dim() == 3, "value needs to be 3 dim.");
  TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  auto opt_sizes = get_opt_sizes(query);
  if (!opt_sizes[2]) {
    throw std::runtime_error("query's third dimension must be regular.");
  }
  // TODO: Add explicit check that verifies query, key and value are the same
// Tensor bt_mha_func(
//     Tensor input, // either of query, key or value in compressed format for
//                   // self-attention
//     Tensor batch_idx, // corresponding batch_idx to input
//     Tensor word_idx, // corresponding word_idx to input
//     at::Tensor in_proj_weight,
//     c10::optional<at::Tensor> in_proj_bias,
//     at::Tensor out_proj_weight_,
//     int64_t head_num,
//     int64_t size_per_head,
//     int64_t valid_word_num) 
  int64_t batch_size = input.size(0);
  int64_t seq_len = input.size(1);
  int64_t embedding_dim = *(opt_sizes[2]);
  // TODO: BLOCKED ON C++ VERSION OF TO_TENSOR_MASK

  Tensor attr_kernel_Q = at::slice(in_proj_weight, 0, 0, embedding_dim).t().contiguous();
  Tensor attr_kernel_K =
      at::slice(in_proj_weight, 0, embedding_dim, 2 * embedding_dim).t().contiguous();
  Tensor attr_kernel_V = at::slice(in_proj_weight, 0, 2 * embedding_dim).t().contiguous();

  Tensor attr_bias_Q = at::slice(*in_proj_bias, 0, 0, embedding_dim);
  Tensor attr_bias_K =
      at::slice(*in_proj_bias, 0, embedding_dim, 2 * embedding_dim);
  Tensor attr_bias_V = at::slice(*in_proj_bias, 0, 2 * embedding_dim);
  Tensor result = torch::empty_like(input);
  Tensor mask_ones =
      torch::ones({batch_size, seq_len, seq_len}, input.options());
  int64_t input_tensor_size = batch_size * head_num * seq_len * size_per_head;
  int64_t attn_tensor_size = batch_size * head_num * seq_len * seq_len;
  int64_t buf_size = input_tensor_size * 13 + attn_tensor_size;
  at::Tensor buf_tensor = torch::zeros({buf_size}, input.options());
  buf_tensor.sub_(1);

  Tensor out_proj_weight = out_proj_weight_.t().contiguous();
  // std::cout << "input.strides(): " << input.strides() << std::endl;
  // std::cout << "attr_kernel_Q.strides(): " << attr_kernel_Q.strides() << std::endl;
  effectivetransformer::bt_mha(
      input.data_ptr<float>(),
      attr_kernel_Q.data_ptr<float>(),
      attr_kernel_K.data_ptr<float>(),
      attr_kernel_V.data_ptr<float>(),
      input.data_ptr<float>(),
      // result.data_ptr<float>(),
      attr_bias_Q.data_ptr<float>(),
      attr_bias_K.data_ptr<float>(),
      attr_bias_V.data_ptr<float>(),
      out_proj_weight.data_ptr<float>(),
      batch_idx.data_ptr<int>(),
      word_idx.data_ptr<int>(),
      mask_ones.data_ptr<float>(),
      batch_size,
      head_num,
      seq_len,
      size_per_head,
      valid_word_num,
      buf_tensor.data<float>());
  // std::cout << "bt_mha buf_tensor.narrow(0, 0, " << input_tensor_size * 3 << "): " << 
  //   buf_tensor.narrow(0, 0, input_tensor_size * 3) << std::endl;
  // return result;
  return buf_tensor.narrow(0, input_tensor_size, input_tensor_size).reshape_as(input);
  // return buf_tensor.narrow(0, 0, input_tensor_size).reshape_as(input);
  // return buf_tensor;
  // return buf_tensor.narrow(0, std::max(attn_tensor_size, input_tensor_size), input_tensor_size).reshape_as(input);
  // return result;
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def(
      "min_mha(int num_heads, int head_dim, float dropout_p, bool training, Tensor query, Tensor key, Tensor value, Tensor in_proj_weight, Tensor? in_proj_bias, float scaling, Tensor out_proj_weight, Tensor out_proj_bias) -> Tensor");
  m.impl("min_mha", NestedTensorKey, &min_mha);

  m.def("exclusive_scan(Tensor input) -> Tensor");
  m.impl("exclusive_scan", c10::DispatchKey::CUDA, &exclusive_scan);

  m.def(
      "compress_bert_input(Tensor input, Tensor mask, Tensor prefix_sum, Tensor result, Tensor batch_idx, Tensor word_idx, int batch_size, int seq_len, int hidden_dim) -> (Tensor, int, int)");
  m.impl("compress_bert_input", c10::DispatchKey::CUDA, &compress_bert_input);

  m.def(
      "restore_bert_output(Tensor result, Tensor input, Tensor batch_idx, Tensor word_idx, int valid_word_num, int seq_len, int hidden_size) -> Tensor");
  m.impl("restore_bert_output", c10::DispatchKey::CUDA, &restore_bert_output);

  m.def(
      "bt_mha_func(Tensor input, Tensor batch_idx, Tensor word_idx, Tensor in_proj_weight, Tensor? in_proj_bias, Tensor out_proj_weight, int head_num, int size_per_head, int valid_word_num) -> Tensor");
  m.impl("bt_mha_func", c10::DispatchKey::CUDA, &bt_mha_func);
}

} // namespace nested_tensor
} // namespace torch
