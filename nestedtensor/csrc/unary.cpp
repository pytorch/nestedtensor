#include <unary.h>

namespace torch {
namespace nested_tensor {

template <class F>
auto unary(F& fn) {
  return [&fn](const THPNestedTensor& self) {
    if (self.is_contiguous()) {
      _BufferNestedTensor cont_self = self.data().right();
      Tensor buffer = cont_self.get_buffer();
      SizeNode nested_size = cont_self.nested_size();
      Tensor result = at::empty({0}, buffer.options());
      fn(result, buffer);
      return THPNestedTensor(
          _BufferNestedTensor(std::move(result), std::move(nested_size)));
    }
    _BufferNestedTensor cont_self = make_contiguous(self.get_structure());
    Tensor buffer = cont_self.get_buffer();
    SizeNode nested_size = cont_self.nested_size();
    Tensor result = at::empty({0}, buffer.options());
    fn(result, buffer);
    return THPNestedTensor(
        _BufferNestedTensor(std::move(result), std::move(nested_size)));
  };
}

template <class F>
auto unary_out(F& fn) {
  return [&fn](const THPNestedTensor& input, THPNestedTensor out) {
    THPNestedTensor result = unary(fn)(input);
    TensorNode& result_node = result.get_structure();
    TensorNode& out_node = out.get_structure();
    apply(
        [](Tensor& out, Tensor& result) { out.copy_(result); },
        out_node,
        result_node);
    return out;
  };
}

template <class F>
auto unary_(F& fn) {
  return [&fn](THPNestedTensor& self) {
    if (self.is_contiguous()) {
      _BufferNestedTensor cont_self = self.data().right();
      Tensor& buffer = cont_self.get_buffer();
      fn(buffer, buffer);
    } else {
      apply(
          [&fn](at::Tensor& tensor) { fn(tensor, tensor); },
          self.get_structure());
    }
    return self;
  };
}

// template <int64_t N, int64_t M, class F>
template <class F>
void add_unary(auto m, auto c, std::string name, F& at_out) {
  m.def(name.c_str(), torch::nested_tensor::unary(at_out));
  m.def(name.c_str(), torch::nested_tensor::unary_out(at_out));
  // py::arg("input"),
  // py::arg("out") = c10::nullopt);
  c.def(name.c_str(), torch::nested_tensor::unary(at_out));
  c.def(
      (name + std::string("_")).c_str(), torch::nested_tensor::unary_(at_out));
}

void add_unary_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  // Functions
  add_unary(m, c, "abs", at::abs_out);
  add_unary(m, c, "acos", at::acos_out);
  add_unary(m, c, "angle", at::angle_out);
  add_unary(m, c, "asin", at::asin_out);
  add_unary(m, c, "atan", at::atan_out);
  add_unary(m, c, "bitwise_not", at::bitwise_not_out);
  add_unary(m, c, "ceil", at::ceil_out);
  add_unary(m, c, "conj", at::conj_out);
  add_unary(m, c, "cos", at::cos_out);
  add_unary(m, c, "cosh", at::cosh_out);
  add_unary(m, c, "digamma", at::digamma_out);
  add_unary(m, c, "erf", at::erf_out);
  add_unary(m, c, "erfc", at::erfc_out);
  add_unary(m, c, "erfinv", at::erfinv_out);
  add_unary(m, c, "exp", at::exp_out);
  add_unary(m, c, "expm1", at::expm1_out);
  add_unary(m, c, "floor", at::floor_out);
  add_unary(m, c, "frac", at::frac_out);
  add_unary(m, c, "imag", at::imag_out);
  add_unary(m, c, "inverse", at::inverse_out);
  add_unary(m, c, "lgamma", at::lgamma_out);
  add_unary(m, c, "log", at::log_out);
  add_unary(m, c, "log10", at::log10_out);
  add_unary(m, c, "log1p", at::log1p_out);
  add_unary(m, c, "log2", at::log2_out);
  add_unary(m, c, "logical_not", at::logical_not_out);
  add_unary(m, c, "neg", at::neg_out);
  add_unary(m, c, "nonzero", at::nonzero_out);
  add_unary(m, c, "real", at::real_out);
  add_unary(m, c, "reciprocal", at::reciprocal_out);
  add_unary(m, c, "round", at::round_out);
  add_unary(m, c, "rsqrt", at::rsqrt_out);
  add_unary(m, c, "sigmoid", at::sigmoid_out);
  add_unary(m, c, "sign", at::sign_out);
  add_unary(m, c, "sin", at::sin_out);
  add_unary(m, c, "sinh", at::sinh_out);
  add_unary(m, c, "sqrt", at::sqrt_out);
  add_unary(m, c, "tan", at::tan_out);
  add_unary(m, c, "tanh", at::tanh_out);
  add_unary(m, c, "trunc", at::trunc_out);
}

} // namespace nested_tensor
} // namespace torch
