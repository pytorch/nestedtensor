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
  TORCH_CHECK(query.dim() == 3, "query needs to be 3 dim.");
  TORCH_CHECK(key.dim() == 3, "key needs to be 3 dim.");
  TORCH_CHECK(value.dim() == 3, "value needs to be 3 dim.");
  TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  int64_t edim = query.size(2);

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

static auto registry =
    torch::RegisterOperators().op("nestedtensor::min_mha", &min_mha);

} // namespace nested_tensor
} // namespace torch
