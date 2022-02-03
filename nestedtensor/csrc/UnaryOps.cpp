#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

// NOTE: Can't reuse dispatch from cos_ to cos_out either, because it requries
// support for at::empty through unary_op_impl
template <class F, F func>
Tensor& NestedTensor_unary_(Tensor& self) {
  apply_nested_tensor([](at::Tensor& tensor) { func(tensor); }, self);
  return self;
}

// NOTE: Missing at::sign_ etc. -> very annoying. not clear why.
template <class F, F func>
Tensor& NestedTensor_unary_method_(Tensor& self) {
  apply_nested_tensor([](at::Tensor& tensor) { (tensor.*func)(); }, self);
  return self;
}

template <class F, F func>
Tensor NestedTensor_unary(const Tensor& self) {
  return map_nested_tensor(
      [](at::Tensor tensor) { return func(tensor); }, self);
}

template <class F, F func>
Tensor& NestedTensor_unary_out(const Tensor& self, Tensor& result) {
  apply_nested_tensor(
      [](Tensor& result, Tensor& self) { func(result, self); }, result, self);
  return result;
}

Tensor& NestedTensor_clamp_(
    Tensor& self,
    const optional<c10::Scalar>& min,
    const optional<c10::Scalar>& max) {
  apply_nested_tensor(
      [min, max](at::Tensor& tensor) { at::clamp_(tensor, min, max); }, self);
  return self;
}

Tensor NestedTensor_clamp(
    const Tensor& self,
    const optional<c10::Scalar>& min,
    const optional<c10::Scalar>& max) {
  return map_nested_tensor(
      [min, max](at::Tensor tensor) { return at::clamp(tensor, min, max); },
      self);
}

Tensor& NestedTensor_clamp_out(
    const Tensor& self,
    const optional<Scalar>& min,
    const optional<Scalar>& max,
    Tensor& result) {
  apply_nested_tensor(
      [min, max](const at::Tensor self, at::Tensor result) {
        at::clamp_out(result, self, min, max);
      },
      self,
      result);
  return result;
}

Tensor& NestedTensor_clamp_min_(Tensor& self, const c10::Scalar& min) {
  apply_nested_tensor(
      [min](at::Tensor& tensor) { at::clamp_min_(tensor, min); }, self);
  return self;
}

Tensor NestedTensor_clamp_min(const Tensor& self, const c10::Scalar& min) {
  return map_nested_tensor(
      [min](at::Tensor tensor) { return at::clamp_min(tensor, min); }, self);
}

Tensor& NestedTensor_clamp_min_out(
    const Tensor& self,
    const c10::Scalar& min,
    Tensor& result) {
  apply_nested_tensor(
      [min](at::Tensor result, const at::Tensor tensor) {
        at::clamp_min_out(result, tensor, min);
      },
      result,
      self);
  return result;
}

Tensor& NestedTensor_clamp_max_(Tensor& self, const c10::Scalar& min) {
  apply_nested_tensor(
      [min](at::Tensor tensor) { at::clamp_max_(tensor, min); }, self);
  return self;
}

Tensor NestedTensor_clamp_max(const Tensor& self, const c10::Scalar& min) {
  return map_nested_tensor(
      [min](at::Tensor tensor) { return at::clamp_max(tensor, min); }, self);
}

Tensor& NestedTensor_clamp_max_out(
    const Tensor& self,
    const Scalar& max,
    Tensor& result) {
  apply_nested_tensor(
      [max](Tensor result, const Tensor tensor) {
        at::clamp_max_out(result, tensor, max);
      },
      result,
      self);
  return result;
}

Tensor& NestedTensor_mvlgamma_(Tensor& self, int64_t p) {
  apply_nested_tensor([p](at::Tensor tensor) { tensor.mvlgamma_(p); }, self);
  return self;
}

Tensor NestedTensor_mvlgamma(const Tensor& self, int64_t p) {
  return map_nested_tensor(
      [p](at::Tensor tensor) { return at::mvlgamma(tensor, p); }, self);
}

#define UNARY_OP_INPLACE_METHOD(NAME)                                     \
  nt_impl(m, #NAME, (NestedTensor_unary<decltype(&at::NAME), at::NAME>)); \
  nt_impl(                                                                \
      m,                                                                  \
      #NAME "_",                                                          \
      (NestedTensor_unary_method_<                                        \
          decltype(&at::Tensor::NAME##_),                                 \
          &at::Tensor::NAME##_>));                                        \
  nt_impl(                                                                \
      m,                                                                  \
      #NAME ".out",                                                       \
      (NestedTensor_unary_out<decltype(&at::NAME##_out), at::NAME##_out>));

#define UNARY_OP(NAME)                                                    \
  nt_impl(m, #NAME, (NestedTensor_unary<decltype(&at::NAME), at::NAME>)); \
  nt_impl(                                                                \
      m,                                                                  \
      #NAME "_",                                                          \
      (NestedTensor_unary_<decltype(&at::NAME##_), at::NAME##_>));        \
  nt_impl(                                                                \
      m,                                                                  \
      #NAME ".out",                                                       \
      (NestedTensor_unary_out<decltype(&at::NAME##_out), at::NAME##_out>));

#define UNARY_OP_NO_OUT(NAME)                                             \
  nt_impl(m, #NAME, (NestedTensor_unary<decltype(&at::NAME), at::NAME>)); \
  nt_impl(                                                                \
      m,                                                                  \
      #NAME "_",                                                          \
      (NestedTensor_unary_<decltype(&at::NAME##_), at::NAME##_>));

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
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
  // UNARY_OP(round);
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
  nt_impl(m, "mvlgamma", NestedTensor_mvlgamma);
  nt_impl(m, "mvlgamma_", NestedTensor_mvlgamma_);

  nt_impl(m, "clamp", NestedTensor_clamp);
  nt_impl(m, "clamp_", NestedTensor_clamp_);
  nt_impl(m, "clamp.out", NestedTensor_clamp_out);

  nt_impl(m, "clamp_min", NestedTensor_clamp_min);
  nt_impl(m, "clamp_min_", NestedTensor_clamp_min_);
  nt_impl(m, "clamp_min.out", NestedTensor_clamp_min_out);

  nt_impl(m, "clamp_max", NestedTensor_clamp_max);
  nt_impl(m, "clamp_max_", NestedTensor_clamp_max_);
  nt_impl(m, "clamp_max.out", NestedTensor_clamp_max_out);
}

} // namespace at
