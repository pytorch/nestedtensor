#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_softmax(
    const Tensor& input,
    const int64_t dim_,
    c10::optional<ScalarType> dtype) {
  int64_t dim = maybe_wrap_dim(dim_, get_dim(input));
  auto input_data = get_nested_tensor_impl(input);
  int64_t nested_dim = input_data->nested_dim();
  TORCH_CHECK(
      dim >= nested_dim,
      "Cannot apply softmax across nested dimensions ",
      std::to_string(dim));
  return map_nested_tensor(
      [dim, nested_dim, dtype](const at::Tensor t) {
        return at::softmax(t, dim - nested_dim, dtype);
      },
      input);
}

Tensor NestedTensor_log_softmax(
    const Tensor& input,
    const int64_t dim_,
    c10::optional<ScalarType> dtype) {
  int64_t dim = maybe_wrap_dim(dim_, get_dim(input));
  auto input_data = get_nested_tensor_impl(input);
  int64_t nested_dim = input_data->nested_dim();
  TORCH_CHECK(
      dim >= nested_dim,
      "Cannot apply log_softmax across nested dimensions ",
      std::to_string(dim));
  return map_nested_tensor(
      [dim, nested_dim, dtype](const at::Tensor t) {
        return at::log_softmax(t, dim - nested_dim, dtype);
      },
      input);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "softmax.int", NestedTensor_softmax);
  nt_impl(m, "log_softmax.int", NestedTensor_log_softmax);
}

} // namespace at
