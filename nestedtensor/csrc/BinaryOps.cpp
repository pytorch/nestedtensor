#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_add_Tensor(
    const Tensor& self_,
    const Tensor& other_,
    const Scalar& alpha) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  // std::cout << "1 get_dim(self): " << get_dim(self) << std::endl;
  // std::cout << "1 get_dim(other): " << get_dim(other) << std::endl;
  // std::cout << "1 is_nested_tensor_impl(self): " << is_nested_tensor_impl(self)
  //           << std::endl;
  // std::cout << "1 is_nested_tensor_impl(other): "
  //           << is_nested_tensor_impl(other) << std::endl;
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other) &&
      get_is_contiguous(self)) {
    // std::cout << "01" << std::endl;
    int64_t self_dim = get_dim(self);
    auto self_opt_sizes = get_opt_sizes(self);
    // std::cout << "self_opt_sizes.size(): " << self_opt_sizes.size() << std::endl;
    // std::cout << "self_dim: " << self_dim << std::endl;
    // std::cout << "other.dim(): " << other.dim() << std::endl;
    // std::cout << "other.size(0): " << other.size(0) << std::endl;
    // if (self_opt_sizes[self_dim - 1]) {
    //   std::cout << "*self_opt_sizes[self_dim - 1]: "
    //             << *self_opt_sizes[self_dim - 1] << std::endl;
    // }
    if (self_opt_sizes[self_dim - 1] && other.dim() == 1 &&
        (*(self_opt_sizes[self_dim - 1])) == other.size(0)) {
      // std::cout << "calling optimized add" << std::endl;
      Tensor self_buffer = get_buffer(self);
      Tensor result_buffer =
          at::add(self_buffer.reshape({-1, other.size(0)}), other)
              .reshape({-1});
      return wrap_buffer(
          std::move(result_buffer),
          get_efficient_nested_size(self),
          get_efficient_nested_stride(self));
    }
  }
  return map_nested_tensor(
      [&alpha](Tensor s, Tensor o) { return at::add(s, o, alpha); },
      self,
      other);
}

Tensor& NestedTensor_add__Tensor(
    Tensor& self_,
    const Tensor& other_,
    const Scalar& alpha) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [&alpha](Tensor& tensor, const Tensor other) {
        tensor.add_(other, alpha);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor& NestedTensor_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  TORCH_CHECK(
      is_nested_tensor_impl(out),
      "NT binary out variant requires NT as out argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(out, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [&alpha](Tensor& self, Tensor& other, Tensor& out) {
        return at::add_out(out, self, other, alpha);
      },
      self,
      other,
      out);
  return out;
}

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

Tensor NestedTensor_floor_divide_Tensor(
    const Tensor& self_,
    const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::floor_divide(s, o); }, self, other);
}

Tensor& NestedTensor_floor_divide__Tensor(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {
        tensor.floor_divide_(other);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor& NestedTensor_floor_divide_out(
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
        return at::floor_divide_out(self, other, out);
      },
      self,
      other,
      out);
  return out;
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

Tensor& NestedTensor_sub_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  TORCH_CHECK(
      is_nested_tensor_impl(out),
      "NT binary out variant requires NT as out argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(out, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [&alpha](Tensor& self, Tensor& other, Tensor& out) {
        return at::sub_out(out, self, other, alpha);
      },
      self,
      other,
      out);
  return out;
}

Tensor NestedTensor_sub_Tensor(
    const Tensor& self_,
    const Tensor& other_,
    const Scalar& alpha) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [&alpha](Tensor s, Tensor o) { return at::sub(s, o, alpha); },
      self,
      other);
}

Tensor& NestedTensor_sub__Tensor(
    Tensor& self_,
    const Tensor& other_,
    const Scalar& alpha) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [&alpha](Tensor& tensor, const Tensor other) {
        tensor.sub_(other, alpha);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor& NestedTensor_remainder__Tensor(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {
        tensor.remainder_(other);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor& NestedTensor_atan2_out(
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
        return at::atan2_out(self, other, out);
      },
      self,
      other,
      out);
  return out;
}

Tensor& NestedTensor_atan2_(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {
        tensor.atan2_(other);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor NestedTensor_atan2(const Tensor& self_, const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::atan2(s, o); }, self, other);
}

Tensor NestedTensor_remainder_Tensor(
    const Tensor& self_,
    const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::remainder(s, o); }, self, other);
}

Tensor& NestedTensor_pow__Tensor(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) {
        tensor.pow_(other);
        return tensor;
      },
      self,
      other);
  return self_;
}

Tensor NestedTensor_pow_Scalar(const Scalar& base, const Tensor& exponent_) {
  Tensor exponent = exponent_;
  return map_nested_tensor(
      [&base](Tensor exponent) { return at::pow(base, exponent); }, exponent);
}

Tensor NestedTensor_pow_Tensor_Tensor(
    const Tensor& self_,
    const Tensor& other_) {
  Tensor self;
  Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return at::pow(s, o); }, self, other);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "add.Tensor", NestedTensor_add_Tensor);
  nt_impl(m, "add_.Tensor", NestedTensor_add__Tensor);
  nt_impl(m, "add.out", NestedTensor_add_out);
  nt_impl(m, "div.Tensor", NestedTensor_div_Tensor);
  nt_impl(m, "div_.Tensor", NestedTensor_div__Tensor);
  nt_impl(m, "div.out", NestedTensor_div_out);
  nt_impl(m, "floor_divide", NestedTensor_floor_divide_Tensor);
  nt_impl(m, "floor_divide_.Tensor", NestedTensor_floor_divide__Tensor);
  nt_impl(m, "floor_divide.out", NestedTensor_floor_divide_out);
  nt_impl(m, "mul.Tensor", NestedTensor_mul_Tensor);
  nt_impl(m, "mul_.Tensor", NestedTensor_mul__Tensor);
  nt_impl(m, "mul.out", NestedTensor_mul_out);
  nt_impl(m, "sub.out", NestedTensor_sub_out);
  nt_impl(m, "sub.Tensor", NestedTensor_sub_Tensor);
  nt_impl(m, "sub_.Tensor", NestedTensor_sub__Tensor);
  nt_impl(m, "remainder_.Tensor", NestedTensor_remainder__Tensor);
  nt_impl(m, "atan2.out", NestedTensor_atan2_out);
  nt_impl(m, "atan2_", NestedTensor_atan2_);
  nt_impl(m, "atan2", NestedTensor_atan2);
  nt_impl(m, "remainder.Tensor", NestedTensor_remainder_Tensor);
  nt_impl(m, "pow_.Tensor", NestedTensor_pow__Tensor);
  nt_impl(m, "pow.Scalar", NestedTensor_pow_Scalar);
  nt_impl(m, "pow.Tensor_Tensor", NestedTensor_pow_Tensor_Tensor);
}

} // namespace at
