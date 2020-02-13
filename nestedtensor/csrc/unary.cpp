#include <unary.h>

namespace torch {
namespace nested_tensor {

template <class F>
auto unary(F& fn) {
  auto new_fn = [&fn](const THPNestedTensor& self) {
    _BufferNestedTensor cont_self = make_contiguous(self.get_structure());
    return THPNestedTensor(_BufferNestedTensor(
        fn(cont_self.get_buffer()), cont_self.nested_size()));
  };
  return new_fn;
}

template <class F>
auto unary_out(F& fn) {
  auto new_fn = [&fn](const THPNestedTensor& input, THPNestedTensor out) {
    THPNestedTensor result = unary(fn)(input);
    TensorNode& result_node = result.get_structure();
    TensorNode& out_node = out.get_structure();
    apply(
        [](Tensor& out, Tensor& result) { out.copy_(result); },
        out_node,
        result_node);
    return out;
  };
  return new_fn;
}

template <class F>
auto unary_(F& fn) {
  auto new_fn = [&fn](THPNestedTensor& self) {
    return unary_out(fn)(self, self);
  };
  return new_fn;
}

// template <int64_t N, int64_t M, class F>
template <class F>
void add_unary(auto m, auto c, std::string name, F& at_function) {
  m.def(name.c_str(), torch::nested_tensor::unary(at_function));
  m.def(name.c_str(), torch::nested_tensor::unary_out(at_function));
  // py::arg("input"),
  // py::arg("out") = c10::nullopt);
  c.def(name.c_str(), torch::nested_tensor::unary(at_function));
  c.def(
      (name + std::string("_")).c_str(),
      torch::nested_tensor::unary_(at_function));
}

void add_unary_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  // Functions
  add_unary(m, c, "abs", at::abs);
  add_unary(m, c, "acos", at::acos);
  add_unary(m, c, "angle", at::angle);
  add_unary(m, c, "asin", at::asin);
  add_unary(m, c, "atan", at::atan);
  add_unary(m, c, "bitwise_not", at::bitwise_not);
  add_unary(m, c, "ceil", at::ceil);
  add_unary(m, c, "conj", at::conj);
  add_unary(m, c, "cos", at::cos);
  add_unary(m, c, "cosh", at::cosh);
  add_unary(m, c, "digamma", at::digamma);
  add_unary(m, c, "erf", at::erf);
  add_unary(m, c, "erfc", at::erfc);
  add_unary(m, c, "erfinv", at::erfinv);
  add_unary(m, c, "exp", at::exp);
  add_unary(m, c, "expm1", at::expm1);
  add_unary(m, c, "floor", at::floor);
  add_unary(m, c, "frac", at::frac);
  add_unary(m, c, "imag", at::imag);
  add_unary(m, c, "inverse", at::inverse);
  add_unary(m, c, "lgamma", at::lgamma);
  add_unary(m, c, "log", at::log);
  add_unary(m, c, "log10", at::log10);
  add_unary(m, c, "log1p", at::log1p);
  add_unary(m, c, "log2", at::log2);
  add_unary(m, c, "logical_not", at::logical_not);
  add_unary(m, c, "neg", at::neg);
  add_unary(m, c, "nonzero", at::nonzero);
  add_unary(m, c, "real", at::real);
  add_unary(m, c, "reciprocal", at::reciprocal);
  add_unary(m, c, "round", at::round);
  add_unary(m, c, "rsqrt", at::rsqrt);
  add_unary(m, c, "sigmoid", at::sigmoid);
  add_unary(m, c, "sign", at::sign);
  add_unary(m, c, "sin", at::sin);
  add_unary(m, c, "sinh", at::sinh);
  add_unary(m, c, "sqrt", at::sqrt);
  add_unary(m, c, "tan", at::tan);
  add_unary(m, c, "tanh", at::tanh);
  add_unary(m, c, "trunc", at::trunc);
}

} // namespace nested_tensor
} // namespace torch
