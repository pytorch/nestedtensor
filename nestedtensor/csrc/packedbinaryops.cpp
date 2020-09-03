#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) { func(tensor, other); },
      self,
      other);
  return self_;
}

template <Tensor (*func)(const Tensor&, Scalar)>
Tensor NestedTensor_binary_scalar(const Tensor& self, Scalar other) {
  return autograd_map_nested_tensor(
      [&other](Tensor self) { return func(self, other); }, self);
}

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return autograd_map_nested_tensor(
      [](Tensor s, Tensor o) { return func(s, o); }, self, other);
}

template <typename S, Tensor (*func)(const Tensor&, const Tensor&, S)>
Tensor NestedTensor_binary(const Tensor& self_, const Tensor& other_, S scalar) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
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
  // at::Tensor self;
  // at::Tensor other;
  // std::tie(self, other) = _expand_other_as(self_, other_);
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
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

// template <Tensor (*func)(const Tensor&, const Tensor&)>
// struct NestedTensorFunction_packed_add
//     : torch::autograd::Function<NestedTensorFunction_packed_add> {
//   static Tensor forward(
//       torch::autograd::AutogradContext* ctx,
//       const Tensor& self,
//       const Tensor& other) {
//     ctx->save_for_backward({self, other});
//     return map_nested_tensor(
//         [](Tensor s, Tensor o) { return func(s, o); }, self, other);
//   }
//   static torch::autograd::variable_list backward(
//       torch::autograd::AutogradContext* ctx,
//       // TODO: To prevent double backward (for now) check that grad_output
//       // doesn't require gradients.
//       torch::autograd::variable_list grad_output) {
//     TORCH_CHECK(
//         grad_output.size() == 1, "Expected grad_output of size 1 for packed binary op.");
//     auto grad = grad_output[0];
//     TORCH_CHECK(
//         !grad.requires_grad(), "addmm does not support double backward.");
//     auto saved_data = ctx->get_saved_variables();
//     auto self = saved_data[0];
//     auto other = saved_data[1];
//     TORCH_CHECK(self.dim() >= 3, "NT self must be at least 3-dim.");
//     TORCH_CHECK(is_nested_tensor_impl(self), "self must be NestedTensor");
//     if (!is_nested_tensor_impl(other)) {
//       TORCH_CHECK(other.dim() >= 2, "T other must be at least 2-dim.");
//       // auto grad_other_nt =
//       //     at::matmul(self.transpose(self.dim() - 2, self.dim() - 1), grad);
//       // TODO: Implement sum over nested dimensions
//       auto grad_other = torch::zeros_like(other);
//       // apply_nested_tensor(
//       //     [&grad_other](at::Tensor& t) { grad_other.add_(t);
//       //     },
//       //     grad_other_nt);
//       apply_nested_tensor(
//           [&grad_other](at::Tensor& s, at::Tensor& g) {
//             grad_other.add_(
//                 at::matmul(s.transpose(s.dim() - 2, s.dim() - 1), g));
//           },
//           self,
//           grad);
//       auto grad_self = at::matmul(grad, other.transpose(0, 1));
//       return {grad_self, grad_other};
//     }
//     TORCH_CHECK(other.dim() >= 3, "NT other must be at least 3-dim.");
//     return {at::matmul(grad, other.transpose(other.dim() - 2, other.dim() - 1)),
//             at::matmul(self.transpose(self.dim() - 2, self.dim() - 1), grad)};
//   }
// };


Tensor NestedTensor_add(
    const Tensor& self_,
    const Tensor& other_,
    Scalar alpha) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return autograd_map_nested_tensor(
      [&alpha](at::Tensor s, at::Tensor o) { return at::add(s, o, alpha); },
      self,
      other);
}

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  // at::Tensor self;
  // at::Tensor other;
  // std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [&](at::Tensor& s, at::Tensor o) { at::native::add_(s, o, alpha); },
      self,
      other);
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
} // namespace at
