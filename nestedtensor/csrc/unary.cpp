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
  auto f = [](at::Tensor tensor) { func(tensor); };
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

#define UNARY_OP(NAME) \
  m.impl_UNBOXED(#NAME, NestedTensor_unary_<decltype(&at:: #NAME), at::cos>); 

//  m.impl_UNBOXED(#NAME_, NestedTensor_unary_<decltype(&at::#NAME_), at::#NAME_>); \
//  m.impl_UNBOXED(#NAME.out, NestedTensor_unary_<decltype(&at::#NAME_out), at::#NAME>_out); 

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {


  // m.impl_UNBOXED("cos_", NestedTensor_unary_<decltype(&at::cos_), at::cos_>);
  // m.impl_UNBOXED("cos", NestedTensor_unary<decltype(&at::cos), at::cos>);
  // m.impl_UNBOXED("cos.out", NestedTensor_unary_out<decltype(&at::cos_out), at::cos_out>);

  UNARY_OP("cos")
  // m.impl_UNBOXED("abs", NestedTensor_unary<decltype(&at::abs), at::abs>);
  m.impl_UNBOXED("acos", NestedTensor_unary<decltype(&at::acos), at::acos>);
  m.impl_UNBOXED("asin", NestedTensor_unary<decltype(&at::asin), at::asin>);
  m.impl_UNBOXED("atan", NestedTensor_unary<decltype(&at::atan), at::atan>);
  m.impl_UNBOXED("ceil", NestedTensor_unary<decltype(&at::ceil), at::ceil>);
  // m.impl_UNBOXED("clamp", NestedTensor_unary<decltype(&at::clamp), at::clamp>);
  // m.impl_UNBOXED("clamp_min", NestedTensor_unary<decltype(&at::clamp_min), at::clamp_min>);
  // m.impl_UNBOXED("clamp_max", NestedTensor_unary<decltype(&at::clamp_max), at::clamp_max>);
  // m.impl_UNBOXED("cos", NestedTensor_unary<decltype(&at::cos), at::cos>);
  m.impl_UNBOXED("cosh", NestedTensor_unary<decltype(&at::cosh), at::cosh>);
  // m.impl_UNBOXED("digamma", NestedTensor_unary<decltype(&at::digamma), at::digamma>);
  m.impl_UNBOXED("erf", NestedTensor_unary<decltype(&at::erf), at::erf>);
  m.impl_UNBOXED("erfc", NestedTensor_unary<decltype(&at::erfc), at::erfc>);
  m.impl_UNBOXED("erfinv", NestedTensor_unary<decltype(&at::erfinv), at::erfinv>);
  m.impl_UNBOXED("exp", NestedTensor_unary<decltype(&at::exp), at::exp>);
  m.impl_UNBOXED("expm1", NestedTensor_unary<decltype(&at::expm1), at::expm1>);
  m.impl_UNBOXED("floor", NestedTensor_unary<decltype(&at::floor), at::floor>);
  // m.impl_UNBOXED("fill", NestedTensor_unary<decltype(&at::fill), at::fill>);
  m.impl_UNBOXED("frac", NestedTensor_unary<decltype(&at::frac), at::frac>);
  m.impl_UNBOXED("lgamma", NestedTensor_unary<decltype(&at::lgamma), at::lgamma>);
  m.impl_UNBOXED("log", NestedTensor_unary<decltype(&at::log), at::log>);
  m.impl_UNBOXED("log10", NestedTensor_unary<decltype(&at::log10), at::log10>);
  m.impl_UNBOXED("log1p", NestedTensor_unary<decltype(&at::log1p), at::log1p>);
  m.impl_UNBOXED("log2", NestedTensor_unary<decltype(&at::log2), at::log2>);
  // m.impl_UNBOXED("mvlgamma", NestedTensor_unary<decltype(&at::mvlgamma), at::mvlgamma>);
  m.impl_UNBOXED("neg", NestedTensor_unary<decltype(&at::neg), at::neg>);
  m.impl_UNBOXED("reciprocal", NestedTensor_unary<decltype(&at::reciprocal), at::reciprocal>);
  m.impl_UNBOXED("round", NestedTensor_unary<decltype(&at::round), at::round>);
  m.impl_UNBOXED("rsqrt", NestedTensor_unary<decltype(&at::rsqrt), at::rsqrt>);
  m.impl_UNBOXED("sigmoid", NestedTensor_unary<decltype(&at::sigmoid), at::sigmoid>);
  m.impl_UNBOXED("sign", NestedTensor_unary<decltype(&at::sign), at::sign>);
  m.impl_UNBOXED("sin", NestedTensor_unary<decltype(&at::sin), at::sin>);
  m.impl_UNBOXED("sinh", NestedTensor_unary<decltype(&at::sinh), at::sinh>);
  m.impl_UNBOXED("sqrt", NestedTensor_unary<decltype(&at::sqrt), at::sqrt>);
  m.impl_UNBOXED("tan", NestedTensor_unary<decltype(&at::tan), at::tan>);
  m.impl_UNBOXED("tanh", NestedTensor_unary<decltype(&at::tanh), at::tanh>);
  m.impl_UNBOXED("trunc", NestedTensor_unary<decltype(&at::trunc), at::trunc>);

}

}
