#include <functions.h>
#include <pybind11/stl.h>
#include <python_functions.h>
#include <torch/extension.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

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

void add_relu(pybind11::module m,
              pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  m.def("relu", relu);
  m.def("relu_", relu_out);
}

void add_dropout(pybind11::module m,
                 pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  m.def("dropout", 
        dropout,
        py::arg("input"),
        py::arg("p") = 0.5,
        py::arg("training") = true,
        py::arg("inplace") = false);
}

void add_conv2d(pybind11::module m,
                pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  m.def("conv2d", 
        conv2d,
        py::arg("input"), 
        py::arg("weight"),
        py::arg("bias") = nullptr,
        py::arg("stride") = std::vector<int64_t>({1, 1}),
        py::arg("padding") = std::vector<int64_t>({0, 0}),
        py::arg("dilation") = std::vector<int64_t>({1, 1}),
        py::arg("groups") = 1);
}

void add_max_pool2d(pybind11::module m,
                    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  m.def("max_pool2d", 
        maxPool2d,
        py::arg("input"), 
        py::arg("kernel_size") = std::vector<int64_t>({}),
        py::arg("stride") = std::vector<int64_t>({}),
        py::arg("padding") = std::vector<int64_t>({0, 0}),
        py::arg("dilation") = std::vector<int64_t>({1, 1}),
        py::arg("return_indices") = false,
        py::arg("ceil_mode") = false);
}

void add_batch_norm(pybind11::module m,
                    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  m.def("batch_norm", 
        batch_norm,
        py::arg("input"),
        py::arg("running_mean"),
        py::arg("running_var"), 
        py::arg("weight") = nullptr,
        py::arg("bias") = nullptr,
        py::arg("training") = false,
        py::arg("momentum") = 0.1,
        py::arg("eps") = 1e-05);
}

void add_cross_entropy(pybind11::module m,
                       pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  m.def("cross_entropy", 
        cross_entropy,
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = nullptr,
        py::arg("size_average") = true,
        py::arg("ignore_index") = -100,
        py::arg("reduce") = true,
        py::arg("reduction") = "mean");
}

void add_interpolate(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
        m.def("interpolate", 
              interpolate,
              py::arg("input"),
              py::arg("size") = nullptr,
              py::arg("scale_factor") = nullptr,
              py::arg("mode") = "nearest",
              py::arg("align_corners") = false);
              //py::arg("recompute_scale_factor") = false);
}

}
} // namespace torch
