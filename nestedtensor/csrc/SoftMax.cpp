#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

#ifdef USE_SUBMODULE
struct NestedTensorFunction_softmax_list
    : torch::autograd::Function<NestedTensorFunction_softmax_list> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& self,
      const int64_t dim,
      c10::optional<ScalarType> dtype) {
    auto self_list = flatten(get_nested_tensor_structure(self));
    auto result_list = _foreach_softmax(self_list, dim, dtype);
    auto result_structure =
        unflatten(get_nested_tensor_structure(self), result_list);
    auto result = wrap_tensor_node(std::move(result_structure));
    ctx->save_for_backward({result, self});
    ctx->saved_data["0"] = dim;
    return result;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(
        grad_output.size() == 1,
        "Expected grad_output of size 1 for packed binary op.");
    auto grad = grad_output[0];
    TORCH_CHECK(
        !grad.requires_grad(), "softmax does not support double backward.");
    std::vector<at::Tensor> saved_data = ctx->get_saved_variables();

    int64_t dim = ctx->saved_data["0"].toInt();
    auto grad_list = flatten(get_nested_tensor_structure(grad));
    auto output = saved_data[0];
    auto input = saved_data[1];
    auto output_list = flatten(get_nested_tensor_structure(output));
    auto input_list = flatten(get_nested_tensor_structure(input));
    auto grad_input_list =
        _foreach_softmax_backward(grad_list, output_list, dim, input_list);
    auto grad_input = wrap_tensor_node(
        unflatten(get_nested_tensor_structure(input), grad_input_list));
    at::Tensor undef;
    return {grad_input, undef, undef};
  }
};
#endif

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

TORCH_LIBRARY_IMPL(aten, AutogradNestedTensor, m) {
  nt_impl(m, "softmax.int", NestedTensor_softmax);
}

} // namespace at
