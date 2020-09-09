#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {
struct NestedTensorFunction_matmul
    : torch::autograd::Function<NestedTensorFunction_matmul> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& self,
      const Tensor& other) {
    ctx->save_for_backward({self, other});
    auto impl_self = get_nested_tensor_impl(self);
    auto structure_self = get_nested_tensor_structure(self);
    if (is_nested_tensor_impl(other)) {
      auto impl_other = get_nested_tensor_impl(other);
      auto structure_other = get_nested_tensor_structure(other);
      if (structure_self.buffer() && structure_other.buffer() &&
          self.dim() == 4 && other.dim() == 4 && impl_self->opt_sizes()[0] &&
          impl_other->opt_sizes()[0] && impl_self->opt_sizes()[1] &&
          impl_other->opt_sizes()[1] && impl_self->opt_sizes()[3] &&
          impl_other->opt_sizes()[2] &&
          (*impl_self->opt_sizes()[0] == *impl_other->opt_sizes()[0]) &&
          (*impl_self->opt_sizes()[1] == *impl_other->opt_sizes()[1]) &&
          (*impl_self->opt_sizes()[3] == *impl_other->opt_sizes()[2])) {
#ifdef TRACEPACKED
        std::cout << "calling packed NT x NT matmul" << std::endl;
#endif
        SizeNode new_nested_size = map(
            [&](c10::List<int64_t> self_size, c10::List<int64_t> other_size) {
              c10::List<int64_t> new_size{
                  self_size[0], self_size[1], other_size[2]};
              return std::move(new_size);
            },
            impl_self->nested_size(),
            impl_other->nested_size());
        auto fn = [](c10::List<int64_t> leaf, int64_t input) {
          return input + leaf[0] * leaf[1] * leaf[2];
        };
        int64_t new_numel = reduce<decltype(fn), int64_t, c10::List<int64_t>>(
            new_nested_size, fn, 0);
        Tensor new_buffer = at::empty({new_numel}, self.options());
        Tensor result =
            wrap_tensor_node(torch::nested_tensor::impl::build_structure(
                std::move(new_buffer), new_nested_size));
        apply_nested_tensor(
            [](at::Tensor& result, at::Tensor self, at::Tensor other) {
              at::matmul_out(result, self, other);
            },
            result,
            self,
            other);
        return result;
      }
      return map_nested_tensor(
          [](Tensor s, Tensor o) { return at::matmul(s, o); }, self, other);
    }
    if (structure_self.buffer()) {
      if (self.dim() == 3 && other.dim() == 2 && impl_self->opt_sizes()[0] &&
          impl_self->opt_sizes()[2] &&
          impl_self->opt_sizes()[self.dim() - 1] ==
              other.size(self.dim() - 2)) {
#ifdef TRACEPACKED
        std::cout << "calling packed NT x T matmul" << std::endl;
#endif
        SizeNode new_nested_size = map(
            [&](c10::List<int64_t> self_size) {
              c10::List<int64_t> new_size{self_size[0], other.size(1)};
              return std::move(new_size);
            },
            impl_self->nested_size());
        return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
            at::matmul(
                (*structure_self.buffer()).reshape({-1, other.size(0)}), other)
                .reshape(-1),
            new_nested_size));
      }
    }
    return map_nested_tensor(
        [&other](Tensor tensor) { return at::matmul(tensor, other); }, self);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(
        grad_output.size() == 1, "Expected grad_output of size 1 for addmm.");
    auto grad = grad_output[0];
    TORCH_CHECK(
        !grad.requires_grad(), "addmm does not support double backward.");
    auto saved_data = ctx->get_saved_variables();
    auto self = saved_data[0];
    auto other = saved_data[1];
    TORCH_CHECK(self.dim() >= 3, "NT self must be at least 3-dim.");
    TORCH_CHECK(is_nested_tensor_impl(self), "self must be NestedTensor");
    if (!is_nested_tensor_impl(other)) {
      TORCH_CHECK(other.dim() >= 2, "T other must be at least 2-dim.");
      // auto grad_other_nt =
      //     at::matmul(self.transpose(self.dim() - 2, self.dim() - 1), grad);
      // TODO: Implement sum over nested dimensions
      auto grad_other = torch::zeros_like(other);
      // apply_nested_tensor(
      //     [&grad_other](at::Tensor& t) { grad_other.add_(t);
      //     },
      //     grad_other_nt);
      apply_nested_tensor(
          [&grad_other](at::Tensor& s, at::Tensor& g) {
            grad_other.add_(
                at::matmul(s.transpose(s.dim() - 2, s.dim() - 1), g));
          },
          self,
          grad);
      auto grad_self = at::matmul(grad, other.transpose(0, 1));
      return {grad_self, grad_other};
    }
    TORCH_CHECK(other.dim() >= 3, "NT other must be at least 3-dim.");
    return {at::matmul(grad, other.transpose(other.dim() - 2, other.dim() - 1)),
            at::matmul(self.transpose(self.dim() - 2, self.dim() - 1), grad)};
  }
};

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  return autograd_map_nested_tensor(
      [](at::Tensor self, at::Tensor other) { return at::matmul(self, other); },
      self,
      other);
  // return NestedTensorFunction_matmul::apply(self, other);
}

Tensor& NestedTensor_matmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  apply_nested_tensor(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        at::matmul_out(result, tensor, other);
      },
      result,
      self,
      other);
  return result;
}

at::Tensor mm_mat1_backward(
    at::Tensor grad,
    at::Tensor other,
    c10::Scalar alpha) {
  return at::mul(at::matmul(grad, other.transpose(0, 1)), alpha);
}

// TODO: Technically this has the wrong semantics and shouldn't accept NTs of
// 3dim, but there's not addmatml
struct NestedTensorFunction_addmm
    : torch::autograd::Function<NestedTensorFunction_addmm> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& self,
      const Tensor& other,
      c10::Scalar alpha,
      c10::Scalar beta) {
    TORCH_CHECK(!is_nested_tensor_impl(input), "input must be Tensor");
    TORCH_CHECK(is_nested_tensor_impl(self), "self must be NestedTensor");
    TORCH_CHECK(!is_nested_tensor_impl(other), "other must be Tensor");
    // TORCH_CHECK(alpha == 1, "alpha must be 1.");
    // TORCH_CHECK(beta == 1, "beta must be 1.");
    auto impl_self = get_nested_tensor_impl(self);
    auto structure_self = get_nested_tensor_structure(self);
    ctx->save_for_backward({input, self, other});
    ctx->saved_data["3"] = alpha;
    ctx->saved_data["4"] = beta;
    if (structure_self.buffer()) {
      if (self.dim() == 3 && other.dim() == 2 && impl_self->opt_sizes()[0] &&
          impl_self->opt_sizes()[2] &&
          impl_self->opt_sizes()[self.dim() - 1] ==
              other.size(self.dim() - 2)) {
#ifdef TRACEPACKED
        std::cout << "calling packed T x NT x T addmm" << std::endl;
#endif
        SizeNode new_nested_size = map(
            [&](c10::List<int64_t> self_size) {
              c10::List<int64_t> new_size{self_size[0], other.size(1)};
              return std::move(new_size);
            },
            impl_self->nested_size());
        return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
            at::addmm(
                input,
                (*structure_self.buffer()).reshape({-1, other.size(0)}),
                other,
                alpha,
                beta)
                .reshape(-1),
            new_nested_size));
      }
    }
    return map_nested_tensor(
        [&](Tensor tensor) {
          return at::addmm(input, tensor, other, alpha, beta);
        },
        self);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(
        grad_output.size() == 1, "Expected grad_output of size 1 for addmm.");
    auto grad = grad_output[0];
    TORCH_CHECK(
        !grad.requires_grad(), "addmm does not support double backward.");
    auto saved_data = ctx->get_saved_variables();
    auto input = saved_data[0];
    auto self = saved_data[1];
    auto other = saved_data[2];
    auto alpha = ctx->saved_data["3"].toScalar();
    auto beta = ctx->saved_data["4"].toScalar();
    auto grad_other_nt = at::mul(at::matmul(self.transpose(1, 2), grad), alpha);
    auto grad_other = torch::zeros_like(other);
    apply_nested_tensor(
        [&grad_other](at::Tensor& t) { grad_other.add_(t); }, grad_other_nt);
    at::Tensor undef;
    return {at::mul(input, beta),
            mm_mat1_backward(grad, other, alpha),
            grad_other,
            undef,
            undef};
  }
};

Tensor NestedTensor_addmm(
    const Tensor& input,
    const Tensor& self,
    const Tensor& other,
    c10::Scalar alpha,
    c10::Scalar beta) {
  // return NestedTensorFunction_addmm::apply(input, self, other, alpha, beta);
  return autograd_map_nested_tensor(
      [&alpha, &beta](at::Tensor input, at::Tensor self, at::Tensor other) {
        return at::addmm(input, self, other, alpha, beta);
      },
      input,
      self,
      other);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  nt_impl(m, "addmm", NestedTensor_addmm);
  nt_impl(m, "matmul", NestedTensor_matmul);
  nt_impl(m, "matmul.out", NestedTensor_matmul_out);
}
} // namespace at
