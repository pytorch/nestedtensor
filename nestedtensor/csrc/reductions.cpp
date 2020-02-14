#include <python_args.h>
#include <reductions.h>
#include <torch/torch.h>

namespace torch {
namespace nested_tensor {

// torch.sum_out
// sum(Tensor input, *, ScalarType? dtype=None)
// sum(Tensor input, DimnameList[1] dim, bool keepdim=False, *, ScalarType?
// dtype=None, Tensor out=None) sum(Tensor input, IntArrayRef[1] dim, bool
// keepdim=False, *, ScalarType? dtype=None, Tensor out=None)
//
// Tensor.sum
// sum(*, ScalarType? dtype=None)
// sum(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)
// sum(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)

template <class F>
auto reduction_full(F& fn) {
  return [&fn](const THPNestedTensor& self, c10::optional<py::object> dtype) {
    c10::optional<at::Scalar> opt_dtype =
        torch::python_args::optional_lift<at::Scalar>(
            torch::python_args::to_scalar, dtype);
    TensorNode result_node =
        map([&fn, &opt_dtype](Tensor data) { return fn(data, opt_dtype); },
            self.get_structure());
    // Will be a list of 0-dim Tensors
    at::Tensor values = stack(flatten(result_node).vec());
    return fn(values, opt_dtype);
  };
}

template <class F>
void add_full_reduction(auto m, auto c, std::string name, F& fn) {
  m.def(name.c_str(), torch::nested_tensor::reduction_full(fn));
  c.def(name.c_str(), torch::nested_tensor::reduction_full(fn));
}

void add_reductions_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  auto fn = [](Tensor& result, c10::optional<ScalarType> dtype) {
    return at::native::sun(result, dtype);
  };
  add_full_reduction(m, c, "sum", fn);
}

} // namespace nested_tensor
} // namespace torch
