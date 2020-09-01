#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_(Tensor& self, const Tensor& other) {
  check_binary_shape(self, other);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) { func(tensor, other); },
      self,
      other);
  return self;
}

template <Tensor (*func)(const Tensor&, Scalar)>
Tensor NestedTensor_binary_scalar(const Tensor& self, Scalar other) {
  return autograd_map_nested_tensor(
      [&other](Tensor self) { return func(self, other); }, self);
}

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other) {
  check_binary_shape(self, other);
  return autograd_map_nested_tensor(
      [](Tensor s, Tensor o) { return func(s, o); }, self, other);
}

template <typename S, Tensor (*func)(const Tensor&, const Tensor&, S)>
Tensor NestedTensor_binary(const Tensor& self, const Tensor& other, S scalar) {
  check_binary_shape(self, other);
  return autograd_map_nested_tensor(
      [&scalar](Tensor self, Tensor other) {
        return func(self, other, scalar);
      },
      self,
      other);
}

template <Tensor& (*func)(Tensor&, const Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
  check_binary_shape(self, other);
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


Tensor NestedTensor_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  check_binary_shape(self, other);
  return autograd_map_nested_tensor(
      [&alpha](at::Tensor s, at::Tensor o) { return at::add(s, o, alpha); },
      self,
      other);
}

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  check_binary_shape(self, other);
  apply_nested_tensor(
      [&](at::Tensor& s, at::Tensor o) { at::native::add_(s, o, alpha); }, self, other);
  return self;
}


#define BINARY_OP(NAME)                                                    \
  nt_impl(m, #NAME ".Tensor", NestedTensor_binary<at::NAME>);              \
  nt_impl(m, #NAME ".Scalar", NestedTensor_binary_scalar<at::NAME>);       \
  nt_impl(m, #NAME "_.Tensor", NestedTensor_binary_<at::native::NAME##_>); \
  nt_impl(m, #NAME ".out", NestedTensor_binary_out<at::NAME##_out>);

// XXX: We need to disable binary ops below autograd between NT and T, because
// in the backwards pass autograd/engine.cpp uses .sizes() which
// doesn't compare between NTs and Ts.
TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  nt_impl(m, "add.Tensor", NestedTensor_add);
  nt_impl(m, "add_.Tensor", NestedTensor_add_);
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
  nt_impl(m, "pow.Tensor_Tensor", NestedTensor_binary<at::pow>);
}
}
