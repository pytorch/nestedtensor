#include <nestedtensor/csrc/creation.h>
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
  TORCH_CHECK(get_dim(query) == 3, "query needs to be 3 dim.");
  TORCH_CHECK(get_dim(key) == 3, "key needs to be 3 dim.");
  TORCH_CHECK(get_dim(value) == 3, "value needs to be 3 dim.");
  TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  auto opt_sizes = get_opt_sizes(query);
  if (!opt_sizes[2]) {
    throw std::runtime_error("query's third dimension must be regular.");
  }
  int64_t edim = *(opt_sizes[2]);

  at::Tensor q, k, v;
  q = at::matmul(
      query,
      at::slice(in_proj_weight, 0, 0, edim).t().contiguous());
  k = at::matmul(
      key,
      at::slice(in_proj_weight, 0, edim, 2 * edim).t().contiguous());
  v = at::matmul(
      value,
      at::slice(in_proj_weight, 0, 2 * edim).t().contiguous());

  q = q + at::slice(*in_proj_bias, 0, 0, edim).contiguous();
  k = k + at::slice(*in_proj_bias, 0, edim, 2 * edim).contiguous();
  v = v + at::slice(*in_proj_bias, 0, 2 * edim).contiguous();

  q = q * torch::tensor(scaling);

  q = q.reshape({*opt_sizes[0], -1, num_heads, head_dim}).transpose(1, 2);
  k = k.reshape({*opt_sizes[0], -1, num_heads, head_dim}).transpose(1, 2);
  v = v.reshape({*opt_sizes[0], -1, num_heads, head_dim}).transpose(1, 2);
  auto attn_output_weights = at::matmul(q, k.transpose(2, 3));
  attn_output_weights = at::softmax(attn_output_weights, -1);
  attn_output_weights = at::dropout(attn_output_weights, dropout_p, training);
  auto attn_output = at::matmul(attn_output_weights, v);
  attn_output = attn_output.transpose(1, 2).reshape({*opt_sizes[0], -1, edim});
  attn_output = at::matmul(attn_output, out_proj_weight.t());
  attn_output = attn_output + out_proj_bias;
  return attn_output;
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def("min_mha(int num_heads, int head_dim, float dropout_p, bool training, Tensor query, Tensor key, Tensor value, Tensor in_proje_weight, Tensor? in_proj_bias, float scaling, Tensor out_proj_weight, Tensor out_proj_bias) -> Tensor", &min_mha);
  m.impl("min_mha", NestedTensorKey, &min_mha);
}

} // namespace nested_tensor
} // namespace torch
