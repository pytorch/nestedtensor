#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

struct NestedTensorFunction_softmax_list
    : torch::autograd::Function<NestedTensorFunction_softmax_list> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& self,
const int64_t dim, c10::optional<ScalarType> dtype) {
    auto self_list = flatten(get_nested_tensor_structure(self));
    auto result_list = _foreach_softmax(self_list);
    ctx->saved_data["0"] = alpha;
    return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
        at::add(get_buffer(self), get_buffer(other)),
        get_nested_tensor_impl(self)->nested_size()));
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto alpha = ctx->saved_data["0"].toScalar();
    TORCH_CHECK(
        grad_output.size() == 1,
        "Expected grad_output of size 1 for packed binary op.");
    auto grad = grad_output[0];
    TORCH_CHECK(
        !grad.requires_grad(), "addmm does not support double backward.");
    at::Tensor undef;
    return {grad, maybe_multiply(grad, alpha), undef};
  }
};

Tensor NestedTensor_softmax(
    const Tensor& input,
    const int64_t dim_,
    c10::optional<ScalarType> dtype) {
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  auto input_data = get_nested_tensor_impl(input);
  int64_t nested_dim = input_data->nested_dim();
  TORCH_CHECK(
      dim >= nested_dim,
      "Cannot apply softmax across nested dimensions ",
      std::to_string(dim));
  return autograd_map_nested_tensor(
      [dim, nested_dim, dtype](const at::Tensor t) {
        return at::softmax(t, dim - nested_dim, dtype);
      },
      input);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  nt_impl(m, "softmax.int", NestedTensor_softmax);
}

} // namespace at
