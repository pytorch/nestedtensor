#include <functions.h>
#include <pybind11/stl.h>
#include <python_args.h>
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

  m.def("relu", 
        [](THPNestedTensor input, 
           c10::optional<bool> inplace) {
             return THPNestedTensor(relu(input.data().contiguous(), inplace));
        },
        py::arg("input"),
        py::arg("inplace") = false);

  m.def(
    "relu_",
    [](THPNestedTensor& input) {
      input = THPNestedTensor(relu_out(input.data()));
      return input;
    },
    py::arg("input"));

  m.def("dropout", 
        [](THPNestedTensor input, 
           c10::optional<double> p, 
           c10::optional<bool> training, 
           c10::optional<bool> inplace) {
             return THPNestedTensor(dropout(input.data().contiguous(), p, training, inplace));
           },
        py::arg("input"),
        py::arg("p") = 0.5,
        py::arg("training") = true,
        py::arg("inplace") = false);

  m.def("conv2d", 
        [](THPNestedTensor input, 
           const at::Tensor weight, 
           c10::optional<at::Tensor> bias, 
           c10::optional<std::vector<int64_t>> stride,
           c10::optional<std::vector<int64_t>> padding,
           c10::optional<std::vector<int64_t>> dilation,
           c10::optional<int64_t> group) {
             return THPNestedTensor(conv2d(input.data().contiguous(), 
                                           weight, 
                                           bias, 
                                           stride, 
                                           padding, 
                                           dilation, 
                                           group));
           },
        py::arg("input"), 
        py::arg("weight"),
        py::arg("bias") = nullptr,
        py::arg("stride") = std::vector<int64_t>({1, 1}),
        py::arg("padding") = std::vector<int64_t>({0, 0}),
        py::arg("dilation") = std::vector<int64_t>({1, 1}),
        py::arg("groups") = 1);

  m.def(
      "max_pool2d",
      [](THPNestedTensor input,
         IAR kernel_size,
         IAR stride,
         IAR padding,
         IAR dilation,
         bool return_indices,
         bool ceil_mode) {
        if (return_indices) {
          throw std::invalid_argument(
              "max_pool2d currently doesn't support returning indices.");
        }
        return THPNestedTensor(max_pool2d(
            input.data().contiguous(),
            kernel_size.extract<2>(),
            stride.extract<2>(),
            padding.extract<2>(),
            dilation.extract<2>(),
            ceil_mode));
      },
      py::arg("input"),
      py::arg("kernel_size"),
      py::arg("stride") = std::vector<int64_t>({}),
      py::arg("padding") = std::vector<int64_t>({0, 0}),
      py::arg("dilation") = std::vector<int64_t>({1, 1}),
      py::arg("return_indices") = false, // TODO Add overload and kernel
      py::arg("ceil_mode") = false);

  m.def(
      "max_pool2d",
      [](THPNestedTensor input,
         IAR kernel_size,
         IAR stride,
         IAR padding,
         IAR dilation,
         bool return_indices,
         bool ceil_mode) {
        if (return_indices) {
          throw std::invalid_argument(
              "max_pool2d currently doesn't support returning indices.");
        }
        return THPNestedTensor(max_pool2d(
            input.data().contiguous(),
            kernel_size.extract<2>(),
            stride.extract<2>(),
            padding.extract<2>(),
            dilation.extract<2>(),
            ceil_mode));
      },
      py::arg("input"),
      py::arg("kernel_size"),
      py::arg("stride") = std::vector<int64_t>({}),
      py::arg("padding") = std::vector<int64_t>({0, 0}),
      py::arg("dilation") = std::vector<int64_t>({1, 1}),
      py::arg("return_indices") = false,
      py::arg("ceil_mode") = false);

  m.def("batch_norm", 
        [](THPNestedTensor input,
           const at::Tensor running_mean,
           const at::Tensor running_var,
           c10::optional<at::Tensor> weight,
           c10::optional<at::Tensor> bias,
           bool training, 
           double momentum,
           double eps){
             return THPNestedTensor(batch_norm(input.data().contiguous(), 
                                               running_mean, 
                                               running_var, 
                                               weight, 
                                               bias, 
                                               training, 
                                               momentum,
                                               eps));
        },
        py::arg("input"),
        py::arg("running_mean"),
        py::arg("running_var"), 
        py::arg("weight") = nullptr,
        py::arg("bias") = nullptr,
        py::arg("training") = false,
        py::arg("momentum") = 0.1,
        py::arg("eps") = 1e-05);

  m.def("cross_entropy", 
        [](THPNestedTensor input,
           THPNestedTensor target,
           c10::optional<at::Tensor> weight,
           c10::optional<bool> size_average, // TODO: use
           c10::optional<int64_t> ignore_index,
           c10::optional<bool> reduce, // TODO: use
           c10::optional<std::string> reduction) {
             return THPNestedTensor(cross_entropy(input.data().contiguous(),
                                                  target.data().contiguous(),
                                                  weight,
                                                  size_average,
                                                  ignore_index,
                                                  reduce,
                                                  reduction));
        },
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = nullptr,
        py::arg("size_average") = true,
        py::arg("ignore_index") = -100,
        py::arg("reduce") = true,
        py::arg("reduction") = "mean");
  
  m.def("interpolate", 
        [](THPNestedTensor input,
           c10::optional<int64_t> size,
           c10::optional<std::vector<double>> scale_factor,
           c10::optional<std::string> mode,
           c10::optional<bool> align_corners,
           c10::optional<bool> recompute_scale_factor) {
             if (size.has_value()) {
               std::vector<int64_t> sz {size.value(), size.value()};
               return THPNestedTensor(interpolate(input.data().contiguous(), 
                                                sz,
                                                scale_factor, 
                                                mode,
                                                align_corners));
             }

             return THPNestedTensor(interpolate(input.data().contiguous(), 
                                                c10::nullopt,
                                                scale_factor, 
                                                mode,
                                                align_corners));
        },
        py::arg("input"),
        py::arg("size") = nullptr,
        py::arg("scale_factor") = nullptr,
        py::arg("mode") = "nearest",
        py::arg("align_corners") = false,
        py::arg("recompute_scale_factor") = false);

  m.def("interpolate", 
        [](THPNestedTensor input,
           c10::optional<std::vector<int64_t>> size,
           c10::optional<std::vector<double>> scale_factor,
           c10::optional<std::string> mode,
           c10::optional<bool> align_corners,
           c10::optional<bool> recompute_scale_factor) {
             if (size.has_value()) {
               return THPNestedTensor(interpolate(input.data().contiguous(), 
                                                  size.value(),
                                                  scale_factor, 
                                                  mode,
                                                  align_corners));
             }

             return THPNestedTensor(interpolate(input.data().contiguous(), 
                                                c10::nullopt,
                                                scale_factor, 
                                                mode,
                                                align_corners));
        },
        py::arg("input"),
        py::arg("size") = nullptr,
        py::arg("scale_factor") = nullptr,
        py::arg("mode") = "nearest",
        py::arg("align_corners") = false,
        py::arg("recompute_scale_factor") = false);
}
}
} // namespace torch
