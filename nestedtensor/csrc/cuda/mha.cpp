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
  TORCH_CHECK(get_nested_dim(query) == 1, "Query nested dim isn't 1.");
  TORCH_CHECK(get_nested_dim(key) == 1, "Key nested dim isn't 1.");
  TORCH_CHECK(get_nested_dim(value) == 1, "Value nested dim isn't 1.");
  // TORCH_CHECK(in_proj_bias, "Input projection bias needs to be defined.");
  // auto opt_sizes = get_opt_sizes(query);
  // if (!opt_sizes[2]) {
  //   throw std::runtime_error("query's third dimension must be regular.");
  // }
  // TODO: Add explicit check that verifies query, key and value are the same
  // auto start = std::chrono::system_clock::now();
  auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  int64_t embedding_dim = head_dim * num_heads; //*(opt_sizes[2]);
  int64_t head_num = num_heads;
  int64_t size_per_head = embedding_dim / head_num;
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);

  at::Tensor packed = at::matmul(query, attr_kernel.t()) + attr_bias;

  at::Tensor packed_padded = to_padded_tensor(packed, 0).contiguous();
  std::vector<at::Tensor> packed_padded_chunks = packed_padded.chunk(3, -1);
  at::Tensor query_buf = packed_padded_chunks[0];
  at::Tensor key_buf = packed_padded_chunks[1];
  at::Tensor val_buf = packed_padded_chunks[2];
  int64_t batch_size = query_buf.size(0);
  int64_t seq_len = query_buf.size(1);

  query_buf = query_buf.reshape({batch_size, seq_len, head_num, size_per_head}).transpose(1, 2);
  key_buf =     key_buf.reshape({batch_size, seq_len, head_num, size_per_head}).transpose(1, 2);
  val_buf =     val_buf.reshape({batch_size, seq_len, head_num, size_per_head}).transpose(1, 2);

  key_buf = key_buf.transpose(2, 3);
  at::Tensor attn_output_weights = at::matmul(query_buf, key_buf).contiguous();

  auto mask_options =
      torch::TensorOptions().dtype(query.dtype()).device(torch::kCUDA);
  at::Tensor input_mask = to_mask(query, 2);
  input_mask = input_mask.to(options);
  at::Tensor attr_mask = input_mask.view({-1, 1, 1, seq_len}).to(mask_options);
  attr_mask = attr_mask * attr_mask.transpose(2, 3);

  if (query.dtype() == torch::kFloat16) {
    nteffectivetransformer::cuda::softmax_kernel_kernelLauncher<c10::Half>(
        attn_output_weights.data_ptr<c10::Half>(),
        attr_mask.data_ptr<c10::Half>(),
        batch_size,
        head_num,
        seq_len,
        (c10::Half)(scaling),
        defaultStream);
  }

  if (query.dtype() == torch::kFloat) {
    nteffectivetransformer::cuda::softmax_kernel_kernelLauncher<float>(
        attn_output_weights.data_ptr<float>(),
        attr_mask.data_ptr<float>(),
        batch_size,
        head_num,
        seq_len,
        (float)(scaling),
        defaultStream);
  }

  auto attn_output = at::matmul(attn_output_weights, val_buf);
  attn_output = attn_output.transpose(1, 2).reshape({batch_size, seq_len, embedding_dim}).contiguous();
  at::Tensor attr_out = from_padded_tensor(attn_output, get_efficient_nested_size(query));
  return at::matmul(attr_out, out_proj_weight.t());
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def(
      "bt_min_mha(int num_heads, int head_dim, float dropout_p, bool training, Tensor query, Tensor key, Tensor value, Tensor attr_kernel, Tensor attr_bias, float scaling, Tensor out_proj_weight, Tensor out_proj_bias) -> Tensor");
  m.impl("bt_min_mha", NestedTensorKey, &bt_min_mha);
}

} // namespace nested_tensor
} // namespace torch
