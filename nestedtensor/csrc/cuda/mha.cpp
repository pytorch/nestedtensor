#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/cuda/attention.h>
#include <nestedtensor/csrc/cuda/bert_transformer_op.h>
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

std::tuple<int64_t, int64_t> compress_bert_input(
    //    Tensor input, // float - (batch_size, seq_len, hidden_dim)
    Tensor mask, // int32 - (batch_size, seq_len)
    Tensor prefix_sum, // int32
    //    Tensor result, // float - (batch_size * num_head * seq_len *
    //    size_per_head)
    Tensor batch_idx, // int32 - (batch_size, seq_len)
    Tensor word_idx, // int32 - (batch_size, seq_len)
    int64_t batch_size,
    int64_t seq_len,
    int64_t hidden_dim) {
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  effectivetransformer::compressBertInput_kernelLauncher(
      // input.data_ptr<float>(),
      mask.data_ptr<int>(),
      prefix_sum.data_ptr<int>(),
      // result.data_ptr<float>(),
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
  return std::make_tuple((int64_t)(valid_word_num), (int64_t)(last_mask));
  // result, (int64_t)(valid_word_num), (int64_t)(last_mask));
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
    at::Tensor input_mask,
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor in_proj_weight,
    c10::optional<at::Tensor> in_proj_bias,
    double scaling,
    at::Tensor out_proj_weight,
    at::Tensor out_proj_bias,
    at::Tensor attr_mask) {
  // TODO: Assert that max seq_len is 1024!
  TORCH_CHECK(query.dim() == 3, "query needs to be 3 dim.");
  TORCH_CHECK(key.dim() == 3, "key needs to be 3 dim.");
  TORCH_CHECK(value.dim() == 3, "value needs to be 3 dim.");
  TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  auto opt_sizes = get_opt_sizes(query);
  if (!opt_sizes[2]) {
    throw std::runtime_error("query's third dimension must be regular.");
  }
  // TODO: Add explicit check that verifies query, key and value are the same
  // Tensor input;
  // Tensor input_mask;
  // std::tie(input, input_mask) = to_tensor_mask(query, 2);
  int64_t batch_size = input_mask.size(0);
  int64_t seq_len = input_mask.size(1);
  int64_t embedding_dim = *(opt_sizes[2]);
  int64_t head_num = num_heads;
  int64_t size_per_head = embedding_dim / head_num;
  int64_t valid_word_num = 1;
  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  // input = input.to(float_options);
  auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  // input_mask = input_mask.to(options);
  Tensor batch_idx = torch::empty({batch_size, seq_len}, options);
  Tensor word_idx = torch::empty({batch_size, seq_len}, options);
  Tensor prefix_sum = exclusive_scan(input_mask);
  int64_t last_mask;
  std::tie(valid_word_num, last_mask) = compress_bert_input(
      // input,
      input_mask,
      prefix_sum,
      // tmptmp,
      batch_idx,
      word_idx,
      batch_size,
      seq_len,
      embedding_dim);
  // std::cout << "get_buffer(query): " << get_buffer(query) << std::endl;
  // std::cout << "tmp: " << tmp.reshape({-1}) << std::endl;
  at::Tensor tmp = get_buffer(query);
  // TODO: BLOCKED ON C++ VERSION OF TO_TENSOR_MASK

  Tensor attr_kernel_Q =
      at::slice(in_proj_weight, 0, 0, embedding_dim).t().contiguous();
  Tensor attr_kernel_K =
      at::slice(in_proj_weight, 0, embedding_dim, 2 * embedding_dim)
          .t()
          .contiguous();
  Tensor attr_kernel_V =
      at::slice(in_proj_weight, 0, 2 * embedding_dim).t().contiguous();

  Tensor attr_bias_Q = at::slice(*in_proj_bias, 0, 0, embedding_dim);
  Tensor attr_bias_K =
      at::slice(*in_proj_bias, 0, embedding_dim, 2 * embedding_dim);
  Tensor attr_bias_V = at::slice(*in_proj_bias, 0, 2 * embedding_dim);
  int64_t input_tensor_size = batch_size * head_num * seq_len * size_per_head;
  int64_t attn_tensor_size = batch_size * head_num * seq_len * seq_len;
  // int64_t buf_size = input_tensor_size * 13 + attn_tensor_size;
  int64_t buf_size = input_tensor_size * 6 + attn_tensor_size;
  at::Tensor buf_tensor = torch::empty({buf_size}, float_options);
  // buf_tensor.sub_(1);

  out_proj_weight = out_proj_weight.t().contiguous();
  // std::cout << "input.strides(): " << input.strides() << std::endl;
  // std::cout << "attr_kernel_Q.strides(): " << attr_kernel_Q.strides() <<
  // std::endl;
  effectivetransformer::bt_mha(
      tmp.data_ptr<float>(),
      attr_kernel_Q.data_ptr<float>(),
      attr_kernel_K.data_ptr<float>(),
      attr_kernel_V.data_ptr<float>(),
      tmp.data_ptr<float>(),
      // result.data_ptr<float>(),
      attr_bias_Q.data_ptr<float>(),
      attr_bias_K.data_ptr<float>(),
      attr_bias_V.data_ptr<float>(),
      out_proj_weight.data_ptr<float>(),
      batch_idx.data_ptr<int>(),
      word_idx.data_ptr<int>(),
      attr_mask.data_ptr<float>(),
      batch_size,
      head_num,
      seq_len,
      size_per_head,
      valid_word_num,
      buf_tensor.data_ptr<float>(),
      (float)(scaling));
  Tensor tmp2 =
      buf_tensor.narrow(0, input_tensor_size, query.numel()).reshape({-1});
  tmp2 = tmp2.contiguous();
  // std::cout << "tmp2: " << tmp2 << std::endl;
  // Tensor result =
  //     torch::ones({batch_size, seq_len, embedding_dim}, float_options);
  // restore_bert_output(
  //     result,
  //     tmp2,
  //     batch_idx,
  //     word_idx,
  //     valid_word_num,
  //     seq_len,
  //     embedding_dim);
  // Tensor result_nt = *nt_from_tensor_mask(result, input_mask, 1);
  // return result_nt;
  return wrap_buffer(std::move(tmp2), get_nested_size(query));
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def("exclusive_scan(Tensor input) -> Tensor");
  m.impl("exclusive_scan", c10::DispatchKey::CUDA, &exclusive_scan);

  // m.def(
  //     "compress_bert_input(Tensor input, Tensor mask, Tensor prefix_sum,
  //     Tensor result, Tensor batch_idx, Tensor word_idx, int batch_size, int
  //     seq_len, int hidden_dim) -> (Tensor, int, int)");
  // m.impl("compress_bert_input", c10::DispatchKey::CUDA,
  // &compress_bert_input);

  m.def(
      "restore_bert_output(Tensor result, Tensor input, Tensor batch_idx, Tensor word_idx, int valid_word_num, int seq_len, int hidden_size) -> Tensor");
  m.impl("restore_bert_output", c10::DispatchKey::CUDA, &restore_bert_output);

  m.def(
      "bt_min_mha(int num_heads, int head_dim, float dropout_p, bool training, Tensor input_mask, Tensor query, Tensor key, Tensor value, Tensor in_proj_weight, Tensor? in_proj_bias, float scaling, Tensor out_proj_weight, Tensor out_proj_bias, Tensor attr_mask) -> Tensor");
  m.impl("bt_min_mha", NestedTensorKey, &bt_min_mha);
}

} // namespace nested_tensor
} // namespace torch
