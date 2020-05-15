#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

// NOTE: Can't reuse dispatch from cos_ to cos_out either, because it requries
// support for empty through unary_op_impl
template <class F, F func>
Tensor& NestedTensor_unary_(Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [](at::Tensor& tensor) { func(tensor); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

template <class F, F func>
Tensor NestedTensor_unary(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([](at::Tensor tensor) { return func(tensor); },
          self_impl->_data.get_structure()));
}

template <class F, F func>
Tensor& NestedTensor_unary_out(Tensor& result, const Tensor& self) {
  auto result_impl = get_nested_tensor_impl(result);
  auto self_impl = get_nested_tensor_impl(self);
  apply([](at::Tensor& result, const at::Tensor tensor)
          { return func(result, tensor); },
      result_impl->_data.get_structure(),
      self_impl->_data.get_structure());
  return result;
}

#define UNARY_OP(NAME)                                                      \
  m.impl_UNBOXED(#NAME, NestedTensor_unary<decltype(&at::NAME), at::NAME>); \
  m.impl_UNBOXED(                                                           \
      #NAME "_", NestedTensor_unary_<decltype(&at::NAME##_), at::NAME##_>); \
  m.impl_UNBOXED(                                                           \
      #NAME ".out",                                                         \
      NestedTensor_unary_out<decltype(&at::NAME##_out), at::NAME##_out>);

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  UNARY_OP(abs);
  UNARY_OP(acos);
  UNARY_OP(asin);
  UNARY_OP(atan);
  UNARY_OP(ceil);
  // UNARY_OP(clamp);
  // UNARY_OP(clamp_min);
  // UNARY_OP(clamp_max);
  UNARY_OP(cos);
  UNARY_OP(cosh);
  // UNARY_OP(digamma);
  UNARY_OP(erf);
  UNARY_OP(erfc);
  // UNARY_OP(erfinv);
  UNARY_OP(exp);
  UNARY_OP(expm1);
  UNARY_OP(floor);
  // UNARY_OP(fill);
  UNARY_OP(frac);
  // UNARY_OP(lgamma);
  UNARY_OP(log);
  UNARY_OP(log10);
  UNARY_OP(log1p);
  UNARY_OP(log2);
  // UNARY_OP(mvlgamma);
  UNARY_OP(neg);
  UNARY_OP(reciprocal);
  UNARY_OP(round);
  UNARY_OP(rsqrt);
  UNARY_OP(sigmoid);
  // UNARY_OP(sign);
  UNARY_OP(sin);
  UNARY_OP(sinh);
  UNARY_OP(sqrt);
  UNARY_OP(tan);
  UNARY_OP(tanh);
  UNARY_OP(trunc);

}

}
