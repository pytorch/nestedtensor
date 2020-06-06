#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_(Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(other)) {
    apply([](Tensor& tensor, const Tensor other) {
          func(tensor, other);
        },
        get_nested_tensor_structure(self),
        get_nested_tensor_structure(other));
    return self;
  }
  apply(
      [&other](Tensor& tensor) { func(tensor, other); },
      get_nested_tensor_structure(self));
  return self;
}

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(other)) {
    return wrap_tensor_node(
        map([](Tensor tensor, Tensor other) { return func(tensor, other); },
            get_nested_tensor_structure(self),
            get_nested_tensor_structure(other)));
  }
  return wrap_tensor_node(
      map([&other](Tensor tensor) { return func(tensor, other); },
          get_nested_tensor_structure(self)));
}

template <typename S, Tensor (*func)(const Tensor&, const Tensor&, S)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other, S scalar) {
  if (is_nested_tensor_impl(other)) {
    return wrap_tensor_node(
        map([&scalar](Tensor tensor, Tensor other) { return func(tensor, other, scalar); },
            get_nested_tensor_structure(self),
            get_nested_tensor_structure(other)));
  }
  return wrap_tensor_node(
      map([&other, &scalar](Tensor tensor) { return func(tensor, other, scalar); },
          get_nested_tensor_structure(self)));
}

template <Tensor& (*func)(Tensor&, const Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  apply(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        return func(result, tensor, other);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self),
      get_nested_tensor_structure(other));
  return result;
}

Tensor& NestedTensor_sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  apply(
      [&alpha](Tensor& tensor, Tensor& other) {
        at::native::sub_(tensor, other, alpha);
      },
      get_nested_tensor_structure(self),
      get_nested_tensor_structure(other));
  return self;
}

Tensor& NestedTensor_sub_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  apply(
      [&alpha](Tensor& result, Tensor& tensor, Tensor& other) {
        return at::sub_out(result, tensor, other, alpha);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self),
      get_nested_tensor_structure(other));
  return result;
}

Tensor& NestedTensor_pow_out_1(Tensor& result, const Tensor& base, const Tensor& exp) {
  apply(
      [](Tensor& result, Tensor& base, Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(base),
      get_nested_tensor_structure(exp));
  return result;
}

Tensor NestedTensor_pow_1(const Tensor& base, const Tensor& exp) {
  return wrap_tensor_node(
      map([](Tensor base, Tensor exp) { return at::pow(base, exp); },
          get_nested_tensor_structure(base),
          get_nested_tensor_structure(exp)));
}

Tensor& NestedTensor_pow__1(Tensor& base, const Tensor& other) {
  return NestedTensor_pow_out_1(base, base, other);
}

Tensor& NestedTensor_pow_out_2(Tensor& result, const Tensor& base, Scalar exp) {
  apply(
      [&exp](Tensor& result, Tensor& base) {
        return at::pow_out(result, base, exp);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(base));
  return result;
}

Tensor NestedTensor_pow_2(const Tensor& base, Scalar exp) {
  return wrap_tensor_node(
      map([exp](Tensor base) { return at::pow(base, exp); },
          get_nested_tensor_structure(base)));
}

Tensor& NestedTensor_pow_out_3(Tensor& result, Scalar base, const Tensor& exp) {
  apply(
      [&base](Tensor& result, Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(exp));
  return result;
}

#define BINARY_OP(NAME)                                                        \
  m.impl_UNBOXED(#NAME ".Tensor", NestedTensor_binary<at::NAME>);              \
  m.impl_UNBOXED(#NAME "_.Tensor", NestedTensor_binary_<at::native::NAME##_>); \
  m.impl_UNBOXED(#NAME ".out", NestedTensor_binary_out<at::NAME##_out>);

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  BINARY_OP(div)
  BINARY_OP(mul)
  BINARY_OP(remainder)

  m.impl_UNBOXED("add.Tensor", NestedTensor_binary<Scalar, at::add>);

  m.impl_UNBOXED("eq.Tensor", NestedTensor_binary<at::eq>);
  m.impl_UNBOXED("ne.Tensor", NestedTensor_binary<at::ne>);

  m.impl_UNBOXED("matmul", NestedTensor_binary<at::matmul>);
  m.impl_UNBOXED("matmul.out", NestedTensor_binary_out<at::matmul_out>);

  m.impl_UNBOXED("atan2", NestedTensor_binary<at::atan2>);
  m.impl_UNBOXED("atan2_", NestedTensor_binary_<at::native::atan2_>);
  m.impl_UNBOXED("atan2.out", NestedTensor_binary_out<at::atan2_out>);

  m.impl_UNBOXED("sub.Tensor", NestedTensor_binary<Scalar, at::sub>);
  m.impl_UNBOXED("sub_.Tensor", NestedTensor_sub_);
  m.impl_UNBOXED("sub.out", NestedTensor_sub_out);

  m.impl_UNBOXED("pow.Tensor_Tensor_out", NestedTensor_pow_out_1);
  m.impl_UNBOXED("pow.Tensor_Tensor", NestedTensor_pow_1);
  m.impl_UNBOXED("pow_.Tensor", NestedTensor_pow__1);
  m.impl_UNBOXED("pow.Tensor_Scalar_out", NestedTensor_pow_out_2);
  m.impl_UNBOXED("pow.Tensor_Scalar", NestedTensor_pow_2);
  m.impl_UNBOXED("pow.Scalar_out", NestedTensor_pow_out_3);
}
}
