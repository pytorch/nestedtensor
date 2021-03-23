#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  return map_nested_tensor(
      [](at::Tensor self, at::Tensor other) { return at::matmul(self, other); },
      self,
      other);
}

Tensor& NestedTensor_matmul_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  apply_nested_tensor(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        at::matmul_out(result, tensor, other);
      },
      result,
      self,
      other);
  return result;
}

Tensor NestedTensor_addmm(
    const Tensor& input,
    const Tensor& self,
    const Tensor& other,
    const c10::Scalar& alpha,
    const c10::Scalar& beta) {
  return map_nested_tensor(
      [&alpha, &beta](at::Tensor input, at::Tensor self, at::Tensor other) {
        return at::addmm(input, self, other, alpha, beta);
      },
      input,
      self,
      other);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "addmm", NestedTensor_addmm);
  nt_impl(m, "matmul", NestedTensor_matmul);
  nt_impl(m, "matmul.out", NestedTensor_matmul_out);
}
} // namespace at
