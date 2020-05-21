#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

// NOTE: Can't reuse dispatch from cos_ to cos_out either, because it requries
// support for at::empty through unary_op_impl
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

Tensor& NestedTensor_clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [min, max](at::Tensor& tensor) { at::clamp_(tensor, min, max); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

Tensor NestedTensor_clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto self_impl = get_nested_tensor_impl(self);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([min, max](at::Tensor tensor) { return at::clamp(tensor, min, max); },
          self_impl->_data.get_structure()));
}

Tensor& NestedTensor_clamp_out(Tensor& result,
                               const Tensor& self,
                               optional<Scalar> min,
                               optional<Scalar> max) {
  auto result_impl = get_nested_tensor_impl(result);
  auto self_impl = get_nested_tensor_impl(self);
  apply([min, max](at::Tensor& result, const at::Tensor tensor)
          { return at::clamp_out(result, tensor, min, max); },
      result_impl->_data.get_structure(),
      self_impl->_data.get_structure());
  return result;
}

Tensor& NestedTensor_clamp_min_(Tensor& self, Scalar min) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [min](at::Tensor& tensor) { at::clamp_min_(tensor, min); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

Tensor NestedTensor_clamp_min(const Tensor& self, Scalar min) {
  auto self_impl = get_nested_tensor_impl(self);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([min](at::Tensor tensor) { return at::clamp_min(tensor, min); },
          self_impl->_data.get_structure()));
}

Tensor& NestedTensor_clamp_min_out(Tensor& result,
                               const Tensor& self,
                               Scalar min) {
  auto result_impl = get_nested_tensor_impl(result);
  auto self_impl = get_nested_tensor_impl(self);
  apply([min](at::Tensor& result, const at::Tensor tensor)
          { return at::clamp_min_out(result, tensor, min); },
      result_impl->_data.get_structure(),
      self_impl->_data.get_structure());
  return result;
}

Tensor& NestedTensor_clamp_max_(Tensor& self, Scalar min) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [min](at::Tensor& tensor) { at::clamp_max_(tensor, min); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

Tensor NestedTensor_clamp_max(const Tensor& self, Scalar min) {
  auto self_impl = get_nested_tensor_impl(self);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([min](at::Tensor tensor) { return at::clamp_max(tensor, min); },
          self_impl->_data.get_structure()));
}

Tensor& NestedTensor_clamp_max_out(Tensor& result,
                               const Tensor& self,
                               Scalar min) {
  auto result_impl = get_nested_tensor_impl(result);
  auto self_impl = get_nested_tensor_impl(self);
  apply([min](at::Tensor& result, const at::Tensor tensor)
          { return at::clamp_max_out(result, tensor, min); },
      result_impl->_data.get_structure(),
      self_impl->_data.get_structure());
  return result;
}

// NOTE: Missing at::lgamma_ -> very annoying
Tensor& NestedTensor_lgamma_(Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [](at::Tensor& tensor) { tensor.lgamma_(); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

// NOTE: Missing at::erfinv_ -> very annoying
Tensor& NestedTensor_erfinv_(Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [](at::Tensor& tensor) { tensor.erfinv_(); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

// NOTE: Missing at::sign_ -> very annoying
Tensor& NestedTensor_sign_(Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [](at::Tensor& tensor) { tensor.sign_(); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

Tensor& NestedTensor_mvlgamma_(Tensor& self, int64_t p) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [p](at::Tensor& tensor) { tensor.mvlgamma_(p); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

Tensor NestedTensor_mvlgamma(const Tensor& self, int64_t p) {
  auto self_impl = get_nested_tensor_impl(self);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([p](at::Tensor tensor) { return at::mvlgamma(tensor, p); },
          self_impl->_data.get_structure()));
}

#define UNARY_OP_NO_INPLACE(NAME)                                                      \
  m.impl_UNBOXED(#NAME, NestedTensor_unary<decltype(&at::NAME), at::NAME>); \
  m.impl_UNBOXED(                                                           \
      #NAME ".out",                                                         \
      NestedTensor_unary_out<decltype(&at::NAME##_out), at::NAME##_out>);

#define UNARY_OP(NAME) \
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

  UNARY_OP_NO_INPLACE(lgamma)
  m.impl_UNBOXED("lgamma_", NestedTensor_lgamma_);

  UNARY_OP_NO_INPLACE(digamma)
  m.impl_UNBOXED(                                                           
      "digamma_", NestedTensor_unary_<decltype(&at::native::digamma_), at::native::digamma_>);

  UNARY_OP_NO_INPLACE(erfinv)
  m.impl_UNBOXED("erfinv_", NestedTensor_erfinv_);

  UNARY_OP_NO_INPLACE(sign)
  m.impl_UNBOXED("sign_", NestedTensor_sign_);

  m.impl_UNBOXED("mvlgamma", NestedTensor_mvlgamma);
  m.impl_UNBOXED("mvlgamma_", NestedTensor_mvlgamma_);

  m.impl_UNBOXED("clamp", NestedTensor_clamp);
  m.impl_UNBOXED("clamp_", NestedTensor_clamp_);
  m.impl_UNBOXED("clamp.out", NestedTensor_clamp_out);

  m.impl_UNBOXED("clamp_min", NestedTensor_clamp_min);
  m.impl_UNBOXED("clamp_min_", NestedTensor_clamp_min_);
  m.impl_UNBOXED("clamp_min.out", NestedTensor_clamp_min_out);

  m.impl_UNBOXED("clamp_max", NestedTensor_clamp_max);
  m.impl_UNBOXED("clamp_max_", NestedTensor_clamp_max_);
  m.impl_UNBOXED("clamp_max.out", NestedTensor_clamp_max_out);

}

}
