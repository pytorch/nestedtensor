#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

Tensor& NestedTensor_sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  check_binary_shape(self, other);
  if (is_nested_tensor_impl(self, other)) {
    torch_check_tensor_shape_matches(self, other);
    apply_nested_tensor(
        [&alpha](Tensor& tensor, Tensor& other) {
          at::native::sub_(tensor, other, alpha);
        },
        self,
        other);
    return self;
  }
  if (is_nested_tensor_impl(self)) {
    torch_check_tensor_shape_matches(self);
    apply_nested_tensor(
        [&other, &alpha](Tensor& self) {
          at::native::sub_(self, other, alpha);
        },
        self);
    return self;
  }
  torch_check_tensor_shape_matches(other);
  apply_nested_tensor(
      [&self, &alpha](Tensor& other) { at::native::sub_(self, other, alpha); },
      other);
  return self;
}

Tensor& NestedTensor_sub_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
  check_binary_shape(self, other);
  is_nested_tensor_impl(result, self, other);
  apply_nested_tensor(
      [&alpha](Tensor& result, Tensor& tensor, Tensor& other) {
        return at::sub_out(result, tensor, other, alpha);
      },
      result,
      self,
      other);
  return result;
}

Tensor& NestedTensor_pow_out_1(
    Tensor& result,
    const Tensor& base,
    const Tensor& exp) {
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
  check_binary_shape(base, exp);
  if (is_nested_tensor_impl(result, base, exp)) {
    torch_check_tensor_shape_matches(result, base, exp);
    apply_nested_tensor(
        [](Tensor& result, Tensor& base, Tensor& exp) {
          at::pow_out(result, base, exp);
        },
        result,
        base,
        exp);
    return result;
  }
  if (is_nested_tensor_impl(result, base)) {
    torch_check_tensor_shape_matches(result, base);
    apply_nested_tensor(
        [&exp](Tensor& result, Tensor& base) {
          at::pow_out(result, base, exp);
        },
        result,
        base);
    return result;
  }
  TORCH_CHECK(
      is_nested_tensor_impl(result, exp),
      "At least one of base or exp needs to be a NestedTensor");
  torch_check_tensor_shape_matches(result, exp);
  apply_nested_tensor(
      [&exp](Tensor& result, Tensor& base) { at::pow_out(result, base, exp); },
      result,
      base);
  return result;
}

Tensor& NestedTensor_pow__1(Tensor& base, const Tensor& other) {
  check_binary_shape(base, other);
  return NestedTensor_pow_out_1(base, base, other);
}

Tensor& NestedTensor_pow_out_2(Tensor& result, const Tensor& base, Scalar exp) {
  apply_nested_tensor(
      [&exp](Tensor& result, Tensor& base) {
        return at::pow_out(result, base, exp);
      },
      result,
      base);
  return result;
}

Tensor NestedTensor_pow_2(const Tensor& base, Scalar exp) {
  return autograd_map_nested_tensor(
      [exp](Tensor base) { return at::pow(base, exp); }, base);
}

Tensor& NestedTensor_pow_out_3(Tensor& result, Scalar base, const Tensor& exp) {
  apply_nested_tensor(
      [&base](Tensor& result, Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      result,
      exp);
  return result;
}

Tensor NestedTensor_pow_3(Scalar base, const Tensor& exp) {
  return autograd_map_nested_tensor(
      [&base](Tensor exp) { return at::pow(base, exp); }, exp);
}

TORCH_LIBRARY_IMPL(aten, AutogradNestedTensor, m) {
  nt_impl(m, "sub_.Tensor", NestedTensor_sub_);
  nt_impl(m, "sub.out", NestedTensor_sub_out);

  nt_impl(m, "pow.Tensor_Tensor_out", NestedTensor_pow_out_1);
  nt_impl(m, "pow_.Tensor", NestedTensor_pow__1);
  nt_impl(m, "pow.Tensor_Scalar_out", NestedTensor_pow_out_2);
  nt_impl(m, "pow.Tensor_Scalar", NestedTensor_pow_2);
  nt_impl(m, "pow.Scalar_out", NestedTensor_pow_out_3);
  nt_impl(m, "pow.Scalar", NestedTensor_pow_3);
}
} // namespace at
