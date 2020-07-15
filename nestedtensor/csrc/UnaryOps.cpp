#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

// NOTE: Can't reuse dispatch from cos_ to cos_out either, because it requries
// support for at::empty through unary_op_impl
template <class F, F func>
Tensor& NestedTensor_unary_(Tensor& self) {
  apply(
      [](at::Tensor& tensor) { func(tensor); },
      get_nested_tensor_structure(self));
  return self;
}

// NOTE: Missing at::sign_ etc. -> very annoying. not clear why.
template <class F, F func>
Tensor& NestedTensor_unary_method_(Tensor& self) {
  apply(
      [](at::Tensor& tensor) { (tensor.*func)(); },
      get_nested_tensor_structure(self));
  return self;
}

template <class F, F func>
Tensor NestedTensor_unary(const Tensor& self) {
  return wrap_tensor_node(
      map([](at::Tensor tensor) { return func(tensor); },
          get_nested_tensor_structure(self)));
}

template <class F, F func>
Tensor& NestedTensor_unary_out(Tensor& result, const Tensor& self) {
  apply(
      [](at::Tensor& result, at::Tensor& tensor) {
        return func(result, tensor);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self));
  return result;
}

Tensor& NestedTensor_clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  apply(
      [min, max](at::Tensor& tensor) { at::clamp_(tensor, min, max); },
      get_nested_tensor_structure(self));
  return self;
}

Tensor NestedTensor_clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return wrap_tensor_node(
      map([min, max](at::Tensor tensor) { return at::clamp(tensor, min, max); },
          get_nested_tensor_structure(self)));
}

Tensor& NestedTensor_clamp_out(Tensor& result,
                               const Tensor& self,
                               optional<Scalar> min,
                               optional<Scalar> max) {
  apply(
      [min, max](at::Tensor result, const at::Tensor tensor) {
        return at::clamp_out(result, tensor, min, max);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self));
  return result;
}

Tensor& NestedTensor_clamp_min_(Tensor& self, Scalar min) {
  apply(
      [min](at::Tensor& tensor) { at::clamp_min_(tensor, min); },
      get_nested_tensor_structure(self));
  return self;
}

Tensor NestedTensor_clamp_min(const Tensor& self, Scalar min) {
  return wrap_tensor_node(
      map([min](at::Tensor tensor) { return at::clamp_min(tensor, min); },
          get_nested_tensor_structure(self)));
}

Tensor& NestedTensor_clamp_min_out(Tensor& result,
                               const Tensor& self,
                               Scalar min) {
  apply([min](at::Tensor result, const at::Tensor tensor)
          { return at::clamp_min_out(result, tensor, min); },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self));
  return result;
}

Tensor& NestedTensor_clamp_max_(Tensor& self, Scalar min) {
  apply([min](at::Tensor tensor) { at::clamp_max_(tensor, min); },
      get_nested_tensor_structure(self));
  return self;
}

Tensor NestedTensor_clamp_max(const Tensor& self, Scalar min) {
  return wrap_tensor_node(
      map([min](at::Tensor tensor) { return at::clamp_max(tensor, min); },
          get_nested_tensor_structure(self)));
}

Tensor& NestedTensor_clamp_max_out(Tensor& result,
                               const Tensor& self,
                               Scalar min) {
  apply([min](at::Tensor result, const at::Tensor tensor)
        { return at::clamp_max_out(result, tensor, min); },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self));
  return result;
}

Tensor& NestedTensor_mvlgamma_(Tensor& self, int64_t p) {
  apply([p](at::Tensor tensor) { tensor.mvlgamma_(p); },
      get_nested_tensor_structure(self));
  return self;
}

Tensor NestedTensor_mvlgamma(const Tensor& self, int64_t p) {
  return wrap_tensor_node(
      map([p](at::Tensor tensor) { return at::mvlgamma(tensor, p); },
          get_nested_tensor_structure(self)));
}

#define UNARY_OP_INPLACE_METHOD(NAME)                                                      \
  m.impl_UNBOXED(#NAME, NestedTensor_unary<decltype(&at::NAME), at::NAME>); \
  m.impl_UNBOXED(                                                           \
      #NAME "_", NestedTensor_unary_method_<decltype(&at::Tensor::NAME##_), &at::Tensor::NAME##_>); \
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

#define UNARY_OP_NO_OUT(NAME)                                               \
  m.impl_UNBOXED(#NAME, NestedTensor_unary<decltype(&at::NAME), at::NAME>); \
  m.impl_UNBOXED(                                                           \
      #NAME "_", NestedTensor_unary_<decltype(&at::NAME##_), at::NAME##_>);

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  UNARY_OP(abs);
  UNARY_OP(acos);
  UNARY_OP(asin);
  UNARY_OP(atan);
  UNARY_OP(ceil);
  UNARY_OP(cos);
  UNARY_OP(cosh);
  UNARY_OP_INPLACE_METHOD(digamma)
  UNARY_OP(erf);
  UNARY_OP(erfc);
  UNARY_OP_INPLACE_METHOD(erfinv)
  UNARY_OP(exp);
  UNARY_OP(expm1);
  UNARY_OP(floor);
  // UNARY_OP(fill);
  UNARY_OP(frac);
  UNARY_OP_INPLACE_METHOD(lgamma)
  UNARY_OP(log);
  UNARY_OP(log10);
  UNARY_OP(log1p);
  UNARY_OP(log2);
  // UNARY_OP(mvlgamma);
  UNARY_OP(neg);
  UNARY_OP(reciprocal);
  UNARY_OP_NO_OUT(relu);
  UNARY_OP(round);
  UNARY_OP(rsqrt);
  UNARY_OP(sigmoid);
  UNARY_OP_INPLACE_METHOD(sign)
  UNARY_OP(sin);
  UNARY_OP(sinh);
  UNARY_OP(sqrt);
  UNARY_OP(tan);
  UNARY_OP(tanh);
  UNARY_OP(trunc);


  // NOTE: mvlgamma doesn't have an out variant? why?
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
