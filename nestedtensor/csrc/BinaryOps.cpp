
#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_div(const Tensor& self_, const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::native::div(s, o); }, self, other);
}

Tensor& NestedTensor_div_(Tensor& self, const Tensor& other);

Tensor& NestedTensor_div_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out);

Tensor NestedTensor_div(const Tensor& self, const Scalar& other);

Tensor& NestedTensor_div_(Tensor& self, const Scalar& other);

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(, "div.Tensor", NestedTensor_div_Tensor);
  nt_impl(, "div_.Tensor", NestedTensor_div__Tensor);
  nt_impl(, "div.out", NestedTensor_div_out);
  nt_impl(, "div.Scalar", NestedTensor_div_Scalar);
  nt_impl(, "div_.Scalar", NestedTensor_div__Scalar);
}

} // namespace at
