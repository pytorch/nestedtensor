#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

std::tuple<Tensor, Tensor, Tensor, Tensor> NestedTensor_embedding_bag(
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
      [&emb_dim](std::vector<int64_t> inp) {
        std::vector<int64_t> new_size;
        new_size.push_back(emb_dim);
        return new_size;
      },
      get_nested_size(indices_));
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::NestedTensor);
  std::tuple<Tensor, Tensor, Tensor, Tensor> emb_outputs = at::embedding_bag(
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

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "embedding_bag", NestedTensor_embedding_bag);
}

} // namespace at
