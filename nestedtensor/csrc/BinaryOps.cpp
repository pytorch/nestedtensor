#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_(Tensor& self, const Tensor& other) {
  auto self_impl = get_nested_tensor_impl(self);
  if (is_nested_tensor_impl(other)) {
    auto other_impl = get_nested_tensor_impl(other);
    apply([](at::Tensor tensor, const at::Tensor other) {
          func(tensor, other);
        },
        self_impl->_data.get_structure(),
        other_impl->_data.get_structure());
    return self;
  }
  apply([&other](at::Tensor tensor) {
        func(tensor, other);
      },
      self_impl->_data.get_structure());
  return self;
}

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other) {
  auto self_impl = get_nested_tensor_impl(self);
  if (is_nested_tensor_impl(other)) {
    auto other_impl = get_nested_tensor_impl(other);
    return at::detail::make_tensor<NestedTensorImpl>(map(
        [](at::Tensor tensor, at::Tensor other) { return func(tensor, other); },
        self_impl->_data.get_structure(),
        other_impl->_data.get_structure()));
  }
  return at::detail::make_tensor<NestedTensorImpl>(map(
      [&other](at::Tensor tensor) { return func(tensor, other); },
      self_impl->_data.get_structure()));
}

template <Tensor& (*func)(Tensor&, const Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto result_impl = get_nested_tensor_impl(result);
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  apply([](at::Tensor& result,
         const at::Tensor& tensor,
         const at::Tensor& other) { return func(result, tensor, other); },
      result_impl->_data.get_structure(),
      self_impl->_data.get_structure(),
      other_impl->_data.get_structure());
  return result;
}

Tensor& NestedTensor_sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  apply([&alpha](at::Tensor tensor, const at::Tensor other) {
        at::native::sub_(tensor, other, alpha);
      },
      self_impl->_data.get_structure(),
      other_impl->_data.get_structure());
  return self;
}

Tensor NestedTensor_sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  return at::detail::make_tensor<NestedTensorImpl>(map(
      [&alpha](at::Tensor tensor, at::Tensor other) {
        return at::sub(tensor, other, alpha);
      },
      self_impl->_data.get_structure(),
      other_impl->_data.get_structure()));
}

Tensor& NestedTensor_sub_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  auto result_impl = get_nested_tensor_impl(result);
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  apply([&alpha](
          at::Tensor& result,
          const at::Tensor& tensor,
          const at::Tensor& other) {
        return at::sub_out(result, tensor, other, alpha);
      },
      result_impl->_data.get_structure(),
      self_impl->_data.get_structure(),
      other_impl->_data.get_structure());
  return result;
}

Tensor& NestedTensor_pow_out_1(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto result_structure = get_nested_tensor_impl(result)->_data.get_structure();
  auto base_structure = get_nested_tensor_impl(base)->_data.get_structure();
  auto exp_structure = get_nested_tensor_impl(exp)->_data.get_structure();
  apply([](at::Tensor& result,
         const at::Tensor& base,
         const at::Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      result_structure,
      base_structure,
      exp_structure);
  return result;
}

Tensor NestedTensor_pow_1(const Tensor& base, const Tensor& exp) {
  auto base_structure = get_nested_tensor_impl(base)->_data.get_structure();
  auto exp_structure = get_nested_tensor_impl(exp)->_data.get_structure();
  return wrap_tensor_node(
      map([](const at::Tensor base,
             const at::Tensor exp) { return at::pow(base, exp); },
          base_structure,
          exp_structure));
}

Tensor& NestedTensor_pow__1(Tensor& base, const Tensor& other) {
  return NestedTensor_pow_out_1(base, base, other);
}

Tensor& NestedTensor_pow_out_2(Tensor& result, const Tensor& base, Scalar exp) {
  auto result_structure = get_nested_tensor_impl(result)->_data.get_structure();
  auto base_structure = get_nested_tensor_impl(base)->_data.get_structure();
  apply([&exp](at::Tensor& result,
         const at::Tensor& base) {
        return at::pow_out(result, base, exp);
      },
      result_structure,
      base_structure);
  return result;
}

Tensor NestedTensor_pow_2(const Tensor& base, Scalar exp) {
  auto base_structure = get_nested_tensor_impl(base)->_data.get_structure();
  return wrap_tensor_node(
      map([exp](const at::Tensor base) { return at::pow(base, exp); },
          base_structure));
}

Tensor& NestedTensor_pow_out_3(Tensor& result, Scalar base, const Tensor& exp) {
  auto result_structure = get_nested_tensor_impl(result)->_data.get_structure();
  auto exp_structure = get_nested_tensor_impl(exp)->_data.get_structure();
  apply([&base](at::Tensor& result,
         const at::Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      result_structure,
      exp_structure);
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

  m.impl_UNBOXED("atan2", NestedTensor_binary<at::atan2>);
  m.impl_UNBOXED("atan2_", NestedTensor_binary_<at::native::atan2_>);
  m.impl_UNBOXED("atan2.out", NestedTensor_binary_out<at::atan2_out>);

  m.impl_UNBOXED("sub.Tensor", NestedTensor_sub);
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
