
#include <unary.h>
namespace torch {
namespace nested_tensor {

template <class F>
auto unary(F& fn) {
  auto new_fn = [&fn](const THPNestedTensor& self) {
    return THPNestedTensor(make_contiguous(
        map([&fn](at::Tensor tensor) { return fn(tensor); },
            self.get_structure())));
  };
  return new_fn;
}

template <class F>
auto unary_out(F& fn) {
  auto new_fn = [&fn](const THPNestedTensor& input, THPNestedTensor out) {
    THPNestedTensor result = unary(fn)(input);
    TensorNode result_node = result.get_structure();
    TensorNode out_node = out.get_structure();
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
void add_unary(
    auto m,
    auto c,
    std::string name,
    F& at_function) {
  m.def(name.c_str(), torch::nested_tensor::unary(at_function));
  m.def(name.c_str(), torch::nested_tensor::unary_out(at_function));
  // py::arg("input"),
  // py::arg("out") = c10::nullopt);
  c.def(name.c_str(), torch::nested_tensor::unary(at_function));
  c.def(
      (name + std::string("_")).c_str(),
      torch::nested_tensor::unary_(at_function));
}

void add_unaries(auto m, auto c) {
  // Functions
  add_unary(m, c, "cos", at::cos);
  add_unary(m, c, "sin", at::sin);
}

} // namespace nested_tensor
} // namespace torch
