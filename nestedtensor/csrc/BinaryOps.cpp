#include <nestedtensor/csrc/BinaryOps.h>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/add.h>
#include <c10/util/Half.h>
#endif

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_add_Tensor(
    const Tensor& self_,
    const Tensor& other_,
    const Scalar& alpha) {
  Tensor self = self_;
  Tensor other = other_;
  if (is_nested_tensor_impl(self) && is_nested_tensor_impl(other)) {
    EfficientSizeNode self_efficient_nested_size =
        get_efficient_nested_size(self);
    EfficientSizeNode other_efficient_nested_size =
        get_efficient_nested_size(other);
    if (efficient_size_matches(
            self_efficient_nested_size, other_efficient_nested_size)) {
      if (get_is_contiguous(self, c10::MemoryFormat::ChannelsLast) &&
          get_is_contiguous(other, c10::MemoryFormat::ChannelsLast)) {
        return wrap_buffer(
            at::add(
                get_buffer(self).view({-1}), get_buffer(other).view({-1})),
            self_efficient_nested_size,
            get_efficient_nested_stride(self));
      }
      if (!get_is_contiguous(self)) {
        self = NestedTensor_contiguous(self);
      }
      if (!get_is_contiguous(other)) {
        other = NestedTensor_contiguous(other);
      }
      return wrap_buffer(
          at::add(
              get_buffer(self).reshape({-1}), get_buffer(other).reshape({-1})),
          self_efficient_nested_size,
          get_efficient_nested_stride(self));
    }
  }
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other)) {
    self = NestedTensor_contiguous(self);
    int64_t self_dim = get_dim(self);
    auto self_opt_sizes = get_opt_sizes(self);
#ifdef WITH_CUDA
    if (self_dim == 4 && other.dim() == 4 &&
        self_opt_sizes[0] &&
        self_opt_sizes[1] &&
        (*self_opt_sizes[1]) == other.size(1) &&
        other.size(0) == 1 &&
        other.size(2) == 1 &&
        other.size(3) == 1 &&
        self.dtype() ==  c10::ScalarType::Half &&
        other.dtype() == c10::ScalarType::Half) {
      other = other.contiguous();
      at::Tensor self_buffer = get_buffer(self);
      Tensor nt_sizes_ =
          get_efficient_nested_size(self).sizes().to(torch::kInt32);
      Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1);
      Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1);
      Tensor nt_sizes_all = nt_sizes_1 * nt_sizes_2;
      std::vector<int> numbers;
      for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
        for (int64_t j = 0; j < *self_opt_sizes[1]; j++) {
          numbers.push_back(nt_sizes_all[i].item<int>());
        }
      }
      at::Tensor numbers_t = torch::tensor(numbers).to(torch::kInt32);
      Tensor nt_sizes_cumsum =
          at::cumsum(numbers_t, 0).to(torch::kInt32).reshape({-1});
      TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor metadata of unexpected dimension.")
      Tensor nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes_cumsum});
      nt_sizes = nt_sizes.to(torch::kCUDA);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      at::Tensor result_buffer = self_buffer.clone();

      c10::Half* self_ptr = self_buffer.data_ptr<c10::Half>();
      c10::Half* other_ptr = other.data_ptr<c10::Half>();
      c10::Half* result_ptr = result_buffer.data_ptr<c10::Half>();
      nested_tensor::cuda::add_scalar_kernelLauncher(
          self_ptr,
          other_ptr,
          result_ptr,
          (int)(*self_opt_sizes[0] * *self_opt_sizes[1]),
          (int)(*self_opt_sizes[0]),
          nt_sizes.data_ptr<int>(),
          defaultStream);
      return wrap_buffer(std::move(result_buffer), get_efficient_nested_size(self),
          get_efficient_nested_stride(self));
    }
#endif
    if (self_opt_sizes[self_dim - 1] && other.dim() == 1 &&
        (*(self_opt_sizes[self_dim - 1])) == other.size(0)) {
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
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [&alpha](Tensor s, Tensor o) {
      return at::add(s, o, alpha); },
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
  Tensor self = self_;
  Tensor other = other_;
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other)) {
    self = NestedTensor_contiguous(self);
    int64_t self_dim = get_dim(self);
    auto self_opt_sizes = get_opt_sizes(self);
#ifdef WITH_CUDA
    if (self_dim == 4 && other.dim() == 4 &&
        self_opt_sizes[0] &&
        self_opt_sizes[1] &&
        (*self_opt_sizes[1]) == other.size(1) &&
        other.size(0) == 1 &&
        other.size(2) == 1 &&
        other.size(3) == 1 &&
        self.dtype() ==  c10::ScalarType::Half &&
        other.dtype() == c10::ScalarType::Half) {
      other = other.contiguous();
      at::Tensor self_buffer = get_buffer(self);
      Tensor nt_sizes_ =
          get_efficient_nested_size(self).sizes().to(torch::kInt32);
      Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1);
      Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1);
      Tensor nt_sizes_all = nt_sizes_1 * nt_sizes_2;
      std::vector<int> numbers;
      for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
        for (int64_t j = 0; j < *self_opt_sizes[1]; j++) {
          numbers.push_back(nt_sizes_all[i].item<int>());
        }
      }
      at::Tensor numbers_t = torch::tensor(numbers).to(torch::kInt32);
      Tensor nt_sizes_cumsum =
          at::cumsum(numbers_t, 0).to(torch::kInt32).reshape({-1});
      TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor metadata of unexpected dimension.")
      Tensor nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes_cumsum});
      nt_sizes = nt_sizes.to(torch::kCUDA);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      at::Tensor result_buffer = self_buffer.clone();

      c10::Half* self_ptr = self_buffer.data_ptr<c10::Half>();
      c10::Half* other_ptr = other.data_ptr<c10::Half>();
      c10::Half* result_ptr = result_buffer.data_ptr<c10::Half>();
      nested_tensor::cuda::mul_scalar_kernelLauncher(
          self_ptr,
          other_ptr,
          result_ptr,
          (int)(*self_opt_sizes[0] * *self_opt_sizes[1]),
          (int)(*self_opt_sizes[0]),
          nt_sizes.data_ptr<int>(),
          defaultStream);
      return wrap_buffer(std::move(result_buffer), get_efficient_nested_size(self),
          get_efficient_nested_stride(self));
    }
#endif
  }
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) {
      return at::mul(s, o); }, self, other);
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
  Tensor self = self_;
  Tensor other = other_;
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other)) {
    self = NestedTensor_contiguous(self);
    int64_t self_dim = get_dim(self);
    auto self_opt_sizes = get_opt_sizes(self);
#ifdef WITH_CUDA
    if (self_dim == 4 && other.dim() == 4 &&
        self_opt_sizes[0] &&
        self_opt_sizes[1] &&
        (*self_opt_sizes[1]) == other.size(1) &&
        other.size(0) == 1 &&
        other.size(2) == 1 &&
        other.size(3) == 1 &&
        self.dtype() ==  c10::ScalarType::Half &&
        other.dtype() == c10::ScalarType::Half) {
      other = other.contiguous();
      at::Tensor self_buffer = get_buffer(self);
      Tensor nt_sizes_ =
          get_efficient_nested_size(self).sizes().to(torch::kInt32);
      Tensor nt_sizes_1 = at::native::narrow(nt_sizes_, 1, 1, 1);
      Tensor nt_sizes_2 = at::native::narrow(nt_sizes_, 1, 2, 1);
      Tensor nt_sizes_all = nt_sizes_1 * nt_sizes_2;
      std::vector<int> numbers;
      for (int64_t i = 0; i < nt_sizes_all.size(0); i++) {
        for (int64_t j = 0; j < *self_opt_sizes[1]; j++) {
          numbers.push_back(nt_sizes_all[i].item<int>());
        }
      }
      at::Tensor numbers_t = torch::tensor(numbers).to(torch::kInt32);
      Tensor nt_sizes_cumsum =
          at::cumsum(numbers_t, 0).to(torch::kInt32).reshape({-1});
      TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor metadata of unexpected dimension.")
      Tensor nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes_cumsum});
      nt_sizes = nt_sizes.to(torch::kCUDA);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      at::Tensor result_buffer = self_buffer.clone();

      c10::Half* self_ptr = self_buffer.data_ptr<c10::Half>();
      c10::Half* other_ptr = other.data_ptr<c10::Half>();
      c10::Half* result_ptr = result_buffer.data_ptr<c10::Half>();
      nested_tensor::cuda::sub_scalar_kernelLauncher(
          self_ptr,
          other_ptr,
          result_ptr,
          (int)(*self_opt_sizes[0] * *self_opt_sizes[1]),
          (int)(*self_opt_sizes[0]),
          nt_sizes.data_ptr<int>(),
          defaultStream);
      return wrap_buffer(std::move(result_buffer), get_efficient_nested_size(self),
          get_efficient_nested_stride(self));
    }
#endif
  }
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [&alpha](Tensor s, Tensor o) {
      return at::sub(s, o, alpha); },
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
