#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  auto nt_impl = get_nested_tensor_impl(self);
  int64_t nested_dim = nt_impl->nested_dim();
  dim = maybe_wrap_dim(dim, nt_impl->dim());
  TORCH_CHECK(
      dim >= nested_dim, "cumsum of nested dimensions is not implemented yet.");
  return map_nested_tensor(
      [nested_dim, dim](at::Tensor tensor) {
        return at::cumsum(tensor, dim - nested_dim);
      },
      self);
}

#define REDUCE_DIM_LIST_FUNC(NAME, FUNC, MSG)                                 \
  Tensor NestedTensor_##NAME(                                                 \
      const Tensor& self,                                                     \
      c10::ArrayRef<int64_t> dims,                                            \
      bool keepdims,                                                          \
      c10::optional<ScalarType> dtype) {                                      \
    auto nt_impl = get_nested_tensor_impl(self);                              \
    int64_t nested_dim = nt_impl->nested_dim();                               \
    std::vector<int64_t> newdims;                                             \
    for (auto dim : dims) {                                                   \
      dim = maybe_wrap_dim(dim, nt_impl->dim());                              \
      TORCH_CHECK(                                                            \
          dim >= nested_dim,                                                  \
          MSG " of nested dimensions is not implemented yet for dimension " + \
              std::to_string(dim));                                           \
      newdims.push_back(dim - nested_dim);                                    \
    }                                                                         \
    return autograd_map_nested_tensor(                                        \
        [nested_dim, newdims, keepdims](at::Tensor tensor) {                  \
          return FUNC(tensor, c10::ArrayRef<int64_t>(newdims), keepdims);     \
        },                                                                    \
        self);                                                                \
  }

REDUCE_DIM_LIST_FUNC(mean_dim, at::mean, "mean");
REDUCE_DIM_LIST_FUNC(sum_dim, at::sum, "sum");
#undef REDUCE_DIM_LIST_FUNC

Tensor NestedTensor_sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::sum(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({0}, *dtype);
    }
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors);
  return at::sum(all_tensor, dtype);
}

Tensor NestedTensor_mean(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::mean(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({0}, *dtype);
    }
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors);
  return at::mean(all_tensor, dtype);
}

Tensor NestedTensor_prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::prod(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({1}, *dtype);
    }
    return at::ones({1});
  }
  auto all_tensor = at::stack(tensors);
  return at::prod(all_tensor, dtype);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  nt_impl(m, "sum", NestedTensor_sum);
  nt_impl(m, "sum.dim_IntList", NestedTensor_sum_dim);
  nt_impl(m, "mean.dim", NestedTensor_mean_dim);
  nt_impl(m, "mean", NestedTensor_mean);
  nt_impl(m, "prod", NestedTensor_prod);
  nt_impl(m, "cumsum", NestedTensor_cumsum);
}

} // namespace at
