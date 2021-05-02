#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/cuda/attention.h>
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

Tensor compress_bert_input(
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
  return result;
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def(
      "min_mha(int num_heads, int head_dim, float dropout_p, bool training, Tensor query, Tensor key, Tensor value, Tensor in_proje_weight, Tensor? in_proj_bias, float scaling, Tensor out_proj_weight, Tensor out_proj_bias) -> Tensor");
  m.impl("min_mha", NestedTensorKey, &min_mha);

  m.def("exclusive_scan(Tensor input) -> Tensor");
  m.impl("exclusive_scan", c10::DispatchKey::CUDA, &exclusive_scan);

  m.def(
      "compress_bert_input(Tensor input, Tensor mask, Tensor prefix_sum, Tensor result, Tensor batch_idx, Tensor word_idx, int batch_size, int seq_len, int hidden_dim) -> Tensor ");
  m.impl("compress_bert_input", c10::DispatchKey::CUDA, &compress_bert_input);
}

} // namespace nested_tensor
} // namespace torch
