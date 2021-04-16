
#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_div_Tensor(const Tensor& self_, const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::div(s, o); }, self, other);
}

Tensor& NestedTensor_div__Tensor(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {
        tensor.div_(other);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor& NestedTensor_div_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  TORCH_CHECK(
      is_nested_tensor_impl(out),
      "NT binary out variant requires NT as out argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(out, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [](Tensor& self, Tensor& other, Tensor& out) {
        return at::div_out(self, other, out);
      },
      self,
      other,
      out);
  return out;
}

Tensor NestedTensor_div_Scalar(const Tensor& self, const Scalar& other) {
  return self;
}

Tensor& NestedTensor_div__Scalar(Tensor& self, const Scalar& other) {
  return self;
}

Tensor NestedTensor_mul_Tensor(const Tensor& self_, const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::mul(s, o); }, self, other);
}

Tensor& NestedTensor_mul__Tensor(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {
        tensor.mul_(other);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor& NestedTensor_mul_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  TORCH_CHECK(
      is_nested_tensor_impl(out),
      "NT binary out variant requires NT as out argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(out, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [](Tensor& self, Tensor& other, Tensor& out) {
        return at::mul_out(self, other, out);
      },
      self,
      other,
      out);
  return out;
}

Tensor NestedTensor_mul_Scalar(const Tensor& self, const Scalar& other) {
  return self;
}

Tensor& NestedTensor_mul__Scalar(Tensor& self, const Scalar& other) {
  return self;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "div.Tensor", NestedTensor_div_Tensor);
  nt_impl(m, "div_.Tensor", NestedTensor_div__Tensor);
  nt_impl(m, "div.out", NestedTensor_div_out);
  nt_impl(m, "div.Scalar", NestedTensor_div_Scalar);
  nt_impl(m, "div_.Scalar", NestedTensor_div__Scalar);
  nt_impl(m, "mul.Tensor", NestedTensor_mul_Tensor);
  nt_impl(m, "mul_.Tensor", NestedTensor_mul__Tensor);
  nt_impl(m, "mul.out", NestedTensor_mul_out);
  nt_impl(m, "mul.Scalar", NestedTensor_mul_Scalar);
  nt_impl(m, "mul_.Scalar", NestedTensor_mul__Scalar);
}

} // namespace at
