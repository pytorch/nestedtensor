#include <functions.h>
#include <pybind11/stl.h>
#include <python_functions.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

void add_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  auto copy_fn = [](THPNestedTensor self,
                    THPNestedTensor source,
                    bool non_blocking = false) {
    return THPNestedTensor(self.data().copy_(source.data()));
  };
  c.def("copy_", copy_fn);

  m.def(
      "squeeze",
      [](THPNestedTensor self,
         c10::optional<int64_t> dim,
         c10::optional<THPNestedTensor> out) {
        if (out) {
          return THPNestedTensor(squeeze(self.data(), dim, out->data()));
        }
        return THPNestedTensor(squeeze(self.data(), dim, c10::nullopt));
      },
      py::arg("self"),
      py::arg("dim") = nullptr,
      py::arg("out") = nullptr);
  c.def(
      "squeeze",
      [](THPNestedTensor self, c10::optional<int64_t> dim) {
        return THPNestedTensor(squeeze(self.data(), dim, c10::nullopt));
      },
      py::arg("dim") = nullptr);
  // NOTE: It is critical that this is passed by referenc!
  // THPNestedTensor is not holding a pointer and a copy won't
  // copy a pointer to underlying storage, but actually cause
  // a deep copy.
  c.def(
      "squeeze_",
      [](THPNestedTensor& self, c10::optional<int64_t> dim) {
        self.data().squeeze_(dim);
        return self;
      },
      py::arg("dim") = nullptr);
}
}
} // namespace torch
