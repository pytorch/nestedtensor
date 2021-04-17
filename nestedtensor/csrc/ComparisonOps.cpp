#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return func(s, o); }, self, other);
}

template <Tensor (*func)(const Tensor&, const Scalar&)>
Tensor NestedTensor_binary_scalar(const Tensor& self, const Scalar& other) {
  return map_nested_tensor(
      [&other](Tensor self) { return func(self, other); }, self);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "eq.Tensor", NestedTensor_binary<at::eq>);
  nt_impl(m, "eq.Scalar", NestedTensor_binary_scalar<at::eq>);
  nt_impl(m, "ne.Tensor", NestedTensor_binary<at::ne>);
  nt_impl(m, "ne.Scalar", NestedTensor_binary_scalar<at::ne>);
  nt_impl(m, "ge.Tensor", NestedTensor_binary<at::ge>);
  nt_impl(m, "ge.Scalar", NestedTensor_binary_scalar<at::ge>);
}
} // namespace at
