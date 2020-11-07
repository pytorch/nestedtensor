#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

std::tuple<Tensor, Tensor, Tensor, Tensor> NestedTensor__embedding_bag(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const c10::optional<Tensor>& per_sample_weights,
    bool include_last_offset) {
  at::Tensor indices = get_buffer(indices_).contiguous();
  int64_t emb_dim = weight.size(1);
  SizeNode output_size = map(
      [&emb_dim](at::Tensor inp) {
        c10::List<int64_t> new_size;
        new_size.push_back(emb_dim);
        return new_size;
      },
      get_nested_tensor_structure(indices_));
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::NestedTensor);
  std::tuple<Tensor, Tensor, Tensor, Tensor> emb_outputs = at::_embedding_bag(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset);
  at::Tensor emb_output_0 = std::get<0>(emb_outputs).reshape({-1});
  auto output = wrap_buffer(std::move(emb_output_0), output_size);
  return std::make_tuple(
      output,
      std::get<1>(emb_outputs),
      std::get<2>(emb_outputs),
      std::get<3>(emb_outputs));
}

Tensor NestedTensor__embedding_bag_dense_backward(
    const Tensor& grad_,
    const Tensor& indices_,
    const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    const Tensor& max_indices_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<Tensor>& per_sample_weights) {
  TORCH_CHECK(is_nested_tensor_impl(grad_), "grad expected to be NestedTensor");
  TORCH_CHECK(
      is_nested_tensor_impl(indices_), "indices expected to be NestedTensor");
  at::Tensor grad = NestedTensor_to_tensor(grad_, c10::nullopt);
  at::Tensor indices = get_buffer(indices_).contiguous();
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::NestedTensor);
  return at::_embedding_bag_dense_backward(
      grad,
      indices,
      offsets,
      offset2bag,
      bag_size_,
      max_indices_,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights);
}

Tensor NestedTensor__embedding_bag_sparse_backward(
    const Tensor& grad_,
    const Tensor& indices_,
    const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<Tensor>& per_sample_weights) {
  std::cout << "ASDF SPARSE" << std::endl;
  TORCH_CHECK(is_nested_tensor_impl(grad_), "grad expected to be NestedTensor");
  TORCH_CHECK(
      is_nested_tensor_impl(indices_), "indices expected to be NestedTensor");
  at::Tensor grad = NestedTensor_to_tensor(grad_, c10::nullopt);
  at::Tensor indices = get_buffer(indices_).contiguous();
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::NestedTensor);
  return at::_embedding_bag_sparse_backward(
      grad,
      indices,
      offsets,
      offset2bag,
      bag_size_,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  m.impl("_embedding_bag", NestedTensor__embedding_bag);
  m.impl(
      "_embedding_bag_dense_backward",
      NestedTensor__embedding_bag_dense_backward);
  m.impl(
      "_embedding_bag_sparse_backward",
      NestedTensor__embedding_bag_sparse_backward);
}

} // namespace at
