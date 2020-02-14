#include <reductions.h>
#include <torch/torch.h>

namespace torch {
namespace nested_tensor {

template <class F>
auto reduction(F& fn) {
  return [&fn](const THPNestedTensor& self) {
    TensorNode result_node = map(
        [&fn](Tensor data) {
          Tensor result;
          return fn(result, data, IntArrayRef({0}), false, c10::nullopt);
        },
        self.get_structure());
    return THPNestedTensor(_ListNestedTensor(std::move(result_node)));
  };
}

template <class F>
void add_reduction(auto m, auto c, std::string name, F& at_out) {
  m.def(name.c_str(), torch::nested_tensor::reduction(at_out));
  // py::arg("input"),
  // py::arg("out") = c10::nullopt);
  c.def(name.c_str(), torch::nested_tensor::reduction(at_out));
}

void add_reductions_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  auto fn = [](Tensor& result,
               const Tensor& self,
               IntArrayRef dim,
               bool keepdim,
               optional<ScalarType> opt_dtype) {
    return at::native::sum_out(result, self, dim, keepdim, opt_dtype);
  };
  add_reduction(m, c, "sum", fn);
}

} // namespace nested_tensor
} // namespace torch
