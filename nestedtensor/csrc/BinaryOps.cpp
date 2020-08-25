#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/library.h>

// TODO: Non-NT argument support

namespace at {

using namespace torch::nested_tensor;

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_(Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self, other)) {
    apply_nested_tensor(
        [](Tensor& tensor, const Tensor other) { func(tensor, other); },
        self,
        other);
    return self;
  }
  if (is_nested_tensor_impl(other)) {
    apply_nested_tensor([&self](Tensor& other) { func(self, other); }, other);
  }
  apply_nested_tensor([&other](Tensor& self) { func(self, other); }, self);
  return self;
}

template <Tensor (*func)(const Tensor&, Scalar)>
Tensor NestedTensor_binary_scalar(const Tensor& self, Scalar other) {
  return map_nested_tensor(
      [&other](Tensor self) { return func(self, other); }, self);
}

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self, other)) {
    return map_nested_tensor(
        [](Tensor self, Tensor other) { return func(self, other); },
        self,
        other);
  }
  if (is_nested_tensor_impl(other)) {
    return map_nested_tensor(
        [&self](Tensor other) { return func(self, other); }, other);
  }
  if (is_packed(self)) {
    auto self_structure = get_nested_tensor_structure(self);
    auto self_impl = get_nested_tensor_impl(self);
    if (other.dim() == 0 || (other.dim() == 1 && other.numel() == 1)) {
#ifdef TRACEPACKED
      std::cout << "calling packed binary NT x T 0-dim / 1-dim 1-numel"
                << typeid(func).name() << std::endl;
#endif
      return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
          func((*self_structure.buffer()), other),
          get_nested_tensor_impl(self)->nested_size()));
    }
  }
  return map_nested_tensor(
      [&other](Tensor self) { return func(self, other); }, self);
}

template <typename S, Tensor (*func)(const Tensor&, const Tensor&, S)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other, S scalar) {
  if (is_nested_tensor_impl(self, other)) {
    return map_nested_tensor(
        [&scalar](Tensor tensor, Tensor other) {
          return func(tensor, other, scalar);
        },
        self,
        other);
  }
  if (is_nested_tensor_impl(other)) {
    return map_nested_tensor(
        [&self, &scalar](Tensor other) { return func(self, other, scalar); },
        other);
  }
  return map_nested_tensor(
      [&other, &scalar](Tensor self) { return func(self, other, scalar); },
      self);
}

template <Tensor& (*func)(Tensor&, const Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  TORCH_CHECK(
      is_nested_tensor_impl(result, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        return func(result, tensor, other);
      },
      result,
      self,
      other);
  return result;
}

Tensor& NestedTensor_sub_(Tensor& self, const Tensor& other, Scalar alpha) {
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
  if (is_nested_tensor_impl(result, base, exp)) {
    torch_check_tensor_shape_matches(result, base, exp);
    apply_nested_tensor(
        [](Tensor& result, Tensor& base, Tensor& exp) {
          return at::pow_out(result, base, exp);
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
          return at::pow_out(result, base, exp);
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
      [&exp](Tensor& result, Tensor& base) {
        return at::pow_out(result, base, exp);
      },
      result,
      base);
  return result;
}

Tensor& NestedTensor_pow__1(Tensor& base, const Tensor& other) {
  is_nested_tensor_impl(base, other);
  return NestedTensor_pow_out_1(base, base, other);
}

Tensor& NestedTensor_pow_out_2(Tensor& result, const Tensor& base, Scalar exp) {
  is_nested_tensor_impl(result, base);
  apply_nested_tensor(
      [&exp](Tensor& result, Tensor& base) {
        return at::pow_out(result, base, exp);
      },
      result,
      base);
  return result;
}

Tensor NestedTensor_pow_2(const Tensor& base, Scalar exp) {
  is_nested_tensor_impl(base);
  return map_nested_tensor(
      [exp](Tensor base) { return at::pow(base, exp); }, base);
}

Tensor& NestedTensor_pow_out_3(Tensor& result, Scalar base, const Tensor& exp) {
  is_nested_tensor_impl(result, exp);
  apply_nested_tensor(
      [&base](Tensor& result, Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      result,
      exp);
  return result;
}

Tensor NestedTensor_pow_3(Scalar base, const Tensor& exp) {
  is_nested_tensor_impl(exp);
  return map_nested_tensor(
      [&base](Tensor exp) { return at::pow(base, exp); }, exp);
}

#define BINARY_OP(NAME)                                                    \
  nt_impl(m, #NAME ".Tensor", NestedTensor_binary<at::NAME>);              \
  nt_impl(m, #NAME "_.Tensor", NestedTensor_binary_<at::native::NAME##_>); \
  nt_impl(m, #NAME ".out", NestedTensor_binary_out<at::NAME##_out>);

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  BINARY_OP(div)
  BINARY_OP(mul)
  BINARY_OP(remainder)

  // floor_divide has an inconsistent signature
  nt_impl(m, "floor_divide", NestedTensor_binary<at::floor_divide>);
  nt_impl(
      m,
      "floor_divide_.Tensor",
      NestedTensor_binary_<at::native::floor_divide_>);
  nt_impl(m, "floor_divide.out", NestedTensor_binary_out<at::floor_divide_out>);

  nt_impl(m, "eq.Tensor", NestedTensor_binary<at::eq>);
  nt_impl(m, "ne.Tensor", NestedTensor_binary<at::ne>);
  nt_impl(m, "eq.Scalar", NestedTensor_binary_scalar<at::eq>);
  nt_impl(m, "ne.Scalar", NestedTensor_binary_scalar<at::ne>);

  nt_impl(m, "atan2", NestedTensor_binary<at::atan2>);
  nt_impl(m, "atan2_", NestedTensor_binary_<at::native::atan2_>);
  nt_impl(m, "atan2.out", NestedTensor_binary_out<at::atan2_out>);

  nt_impl(m, "sub.Tensor", (NestedTensor_binary<Scalar, at::sub>));
  nt_impl(m, "sub_.Tensor", NestedTensor_sub_);
  nt_impl(m, "sub.out", NestedTensor_sub_out);

  nt_impl(m, "pow.Tensor_Tensor_out", NestedTensor_pow_out_1);
  nt_impl(m, "pow.Tensor_Tensor", NestedTensor_binary<at::pow>);
  nt_impl(m, "pow_.Tensor", NestedTensor_pow__1);
  nt_impl(m, "pow.Tensor_Scalar_out", NestedTensor_pow_out_2);
  nt_impl(m, "pow.Tensor_Scalar", NestedTensor_pow_2);
  nt_impl(m, "pow.Scalar_out", NestedTensor_pow_out_3);
  nt_impl(m, "pow.Scalar", NestedTensor_pow_3);
}
} // namespace at
