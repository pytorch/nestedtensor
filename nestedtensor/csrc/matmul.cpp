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
//     auto impl_other = get_nested_tensor_impl(other);
//     auto structure_other = get_nested_tensor_structure(other);
//     if (structure_self.buffer() && structure_other.buffer() &&
//         self.dim() == 4 && other.dim() == 4 && impl_self->opt_sizes()[0] &&
//         impl_other->opt_sizes()[0] && impl_self->opt_sizes()[1] &&
//         impl_other->opt_sizes()[1] && impl_self->opt_sizes()[3] &&
//         impl_other->opt_sizes()[2] &&
//         (*impl_self->opt_sizes()[0] == *impl_other->opt_sizes()[0]) &&
//         (*impl_self->opt_sizes()[1] == *impl_other->opt_sizes()[1]) &&
//         (*impl_self->opt_sizes()[3] == *impl_other->opt_sizes()[2])) {
// #ifdef TRACEPACKED
//       std::cout << "calling packed NT x NT matmul" << std::endl;
// #endif
//       SizeNode new_nested_size = map(
//           [&](c10::List<int64_t> self_size, c10::List<int64_t> other_size) {
//             c10::List<int64_t> new_size{
//                 self_size[0], self_size[1], other_size[2]};
//             return std::move(new_size);
//           },
//           impl_self->nested_size(),
//           impl_other->nested_size());
//       auto fn = [](c10::List<int64_t> leaf, int64_t input) {
//         return input + leaf[0] * leaf[1] * leaf[2];
//       };
//       int64_t new_numel = reduce<decltype(fn), int64_t, c10::List<int64_t>>(
//           new_nested_size, fn, 0);
//       // Tensor new_buffer = at::empty({new_numel}, self.options());
//       // Tensor result =
//       //     wrap_tensor_node(torch::nested_tensor::impl::build_structure(
//       //         std::move(new_buffer), new_nested_size));
//       return map_nested_tensor(
//           [](//at::Tensor& result,
//              at::Tensor self,
//              at::Tensor other) { at::matmul(self, other); },
//  //         result,
//           self,
//           other);
// //      return result;
//     }
    return map_nested_tensor(
        [](Tensor s, Tensor o) { return at::matmul(s, o); },
        self,
        other);
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
