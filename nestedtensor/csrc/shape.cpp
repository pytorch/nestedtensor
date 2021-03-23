#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_view(const Tensor& self, IntArrayRef size) {
  auto self_data = get_nested_tensor_impl(self);
  TORCH_CHECK(
      int64_t(size.size()) > self_data->nested_dim(),
      "view cannot be exclusive to nested dimensions.");
  for (int64_t i = 0; i < self_data->nested_dim(); i++) {
    if (size[i] >= 0) {
      throw std::runtime_error(
          "Cannot view explicitly along irregular dimension " +
          std::to_string(i) + ". Please use -1 as a placeholder.");
    }
  }
  int64_t nested_dim = self_data->nested_dim();
  std::vector<int64_t> target_shape;
  for (int64_t i = nested_dim; i < int64_t(size.size()); i++) {
    target_shape.push_back(size[i]);
  }
  // TODO: Potential use for packed view, but requires custom backward.
  return map_nested_tensor(
      [target_shape](const at::Tensor t) {
        return at::native::view(t, IntArrayRef(target_shape));
      },
      self);
}

Tensor NestedTensor_reshape(const Tensor& self, IntArrayRef size) {
  auto self_data = get_nested_tensor_impl(self);
  TORCH_CHECK(
      int64_t(size.size()) > self_data->nested_dim(),
      "Reshape cannot be exclusive to nested dimensions.");
  for (int64_t i = 0; i < self_data->nested_dim(); i++) {
    if (size[i] >= 0) {
      throw std::runtime_error(
          "Cannot reshape explicitly along irregular dimension " +
          std::to_string(i) + ". Please use -1 as a placeholder.");
    }
  }
  int64_t nested_dim = self_data->nested_dim();
  std::vector<int64_t> target_shape;
  for (int64_t i = nested_dim; i < int64_t(size.size()); i++) {
    target_shape.push_back(size[i]);
  }
  // TODO: Potential use for packed reshape, but requires custom backward.
  return map_nested_tensor(
      [target_shape](const at::Tensor t) {
        return at::reshape(t, IntArrayRef(target_shape));
      },
      self);
}

Tensor NestedTensor_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto self_data = get_nested_tensor_impl(self);
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }
  int64_t nested_dim = self_data->nested_dim();
  TORCH_CHECK(
      dim0 >= nested_dim && dim1 >= nested_dim,
      "Transposition of nested dimensions is not implemented yet.");
  // TODO: Potential use for packed transpose, but requires custom backward.
  return map_nested_tensor(
      [dim0, dim1, nested_dim](const at::Tensor t) {
        return at::transpose(t, dim0 - nested_dim, dim1 - nested_dim);
      },
      self);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "reshape", NestedTensor_reshape);
  nt_impl(m, "view", NestedTensor_view);
  nt_impl(m, "transpose.int", NestedTensor_transpose);
}

} // namespace at
