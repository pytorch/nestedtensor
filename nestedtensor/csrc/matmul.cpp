#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {
Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  AutoGradMode autogradmode(false);
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
            std::cout << "self_size: " << self_size[0] << " " << self_size[1]
                      << " " << self_size[2] << std::endl;
            std::cout << "other_size: " << other_size[0] << " " << other_size[1]
                      << " " << other_size[2] << std::endl;
            c10::List<int64_t> new_size{
                self_size[0], self_size[1], other_size[2]};
            std::cout << "new_size: " << new_size[0] << " " << new_size[1]
                      << " " << new_size[2] << std::endl;
            return std::move(new_size);
          },
          impl_self->nested_size(),
          impl_other->nested_size());
      auto self_buffer = *structure_self.buffer();
      auto other_buffer = *structure_other.buffer();
      self_buffer = self_buffer.reshape({self.size(1), -1, self.size(3)});
      other_buffer = other_buffer.reshape({self.size(1), other.size(2), -1});
      std::cout << self_buffer.sizes() << std::endl;
      std::cout << other_buffer.sizes() << std::endl;
      auto result_buffer = at::bmm(self_buffer, other_buffer);
      std::cout << result_buffer.sizes() << std::endl;
      result_buffer = result_buffer.reshape({-1});
      auto rr = map_nested_tensor(
          [](Tensor s, Tensor o) { return at::matmul(s, o); }, self, other);
      std::cout << "rr.numel(): " << rr.numel() << std::endl;
      apply_nested_tensor(
          [](at::Tensor a) {
            std::cout << "a.sizes(): " << a.sizes() << std::endl;
          },
          rr);
      return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
          std::move(result_buffer), new_nested_size));
      // auto fn = [](c10::List<int64_t> leaf, int64_t input) {
      //   return input + leaf[0] * leaf[1] * leaf[2];
      // };
      // int64_t new_numel = reduce<decltype(fn), int64_t, c10::List<int64_t>>(
      //     new_nested_size, fn, 0);
      // Tensor new_buffer = at::empty({new_numel}, self.options());
      // Tensor result =
      //     wrap_tensor_node(torch::nested_tensor::impl::build_structure(
      //         std::move(new_buffer), new_nested_size));
      //        return map_nested_tensor(
      //            [](//at::Tensor& result,
      //               at::Tensor self,
      //               at::Tensor other) { at::matmul(self, other); },
      //   //         result,
      //            self,
      //            other);
      //  //      return result;
    }
    return map_nested_tensor(
        [](Tensor s, Tensor o) { return at::matmul(s, o); }, self, other);
  }
  if (structure_self.buffer()) {
    if (self.dim() == 3 && other.dim() == 2 && impl_self->opt_sizes()[0] &&
        impl_self->opt_sizes()[2] &&
        impl_self->opt_sizes()[self.dim() - 1] == other.size(self.dim() - 2)) {
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

Tensor& NestedTensor_matmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  AutoGradMode autogradmode(false);
  apply_nested_tensor(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        return at::matmul_out(result, tensor, other);
      },
      result,
      self,
      other);
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  // nt_impl(m, "matmul", no_bw(TORCH_FN(NestedTensor_matmul);
  nt_impl(m, "matmul", NestedTensor_matmul);
  nt_impl(m, "matmul.out", NestedTensor_matmul_out);
}
} // namespace at
