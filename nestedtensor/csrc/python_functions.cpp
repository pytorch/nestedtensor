#include <nestedtensor/csrc/functions.h>	
#include <nestedtensor/csrc/python_args.h>	
#include <nestedtensor/csrc/python_functions.h>	
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <pybind11/stl.h>	
#include <torch/extension.h>	

using namespace torch::nn;	
namespace F = torch::nn::functional;	

namespace torch {	
namespace nested_tensor {	

NestedTensor interpolate(NestedTensor& input,
                         c10::optional<std::vector<std::vector<int64_t>>> size,
                         c10::optional<at::ArrayRef<double>> scale_factor,
                         c10::optional<std::string> mode,
                         c10::optional<bool> align_corners) {
                         //bool recompute_scale_factor) { // TODO: use
    F::InterpolateFuncOptions::mode_t int_mode;
    if (mode.value() == "nearest" || mode.value() == "none") {
        int_mode = torch::kNearest;
    } else if (mode.value()== "trilinear") {
        int_mode = torch::kTrilinear;
    } else if (mode.value()== "linear") {
        int_mode = torch::kLinear;
    } else if (mode.value()== "bicubic") {
        int_mode = torch::kBicubic;
    } else if (mode.value()== "area") {
        int_mode = torch::kArea;
    } else if (mode.value()== "bilinear") {
        int_mode = torch::kBilinear;
    }  else {
        throw std::runtime_error("Unexpected mode for interpolate: " + mode.value());
    }

    TensorNode input_structure = input.get_structure();
    auto options = F::InterpolateFuncOptions().mode(int_mode);
    if (align_corners.has_value()) {
      options.align_corners() = align_corners.value();
    }

    // Either scale factor or size can be passed
    if (scale_factor.has_value()) {
      options = options.scale_factor(scale_factor.value().vec());
      TensorNode res = map(
        [&options](at::Tensor input_tensor) {
          return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
        },
        input_structure);
      return NestedTensor(std::move(res));
    }

    // Get input leaves count
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + 1;
    };
    auto leaves_count = reduce<decltype(fn), int64_t, at::Tensor>(input.get_structure(), fn, 0);

    if (size.has_value()) {
      // There can be either 1 size for all tensor or an individual size value per tensor
      if (size.value().size() != 1 && size.value().size() != leaves_count) {
        throw std::runtime_error( "Interpolate has to take either 1 size tuple or same amount as leaves in Nested Tensor.");
      }

      if (size.value().size() == 1) {
        TensorNode res = map(
          [&options, &size](at::Tensor input_tensor) {
            options = options.size(size.value()[0]);
            return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
          },
          input_structure);
        return NestedTensor(std::move(res));
      } else {
        int size_i = 0;
        TensorNode res = map(
            [&options, &size_i, &size](at::Tensor input_tensor) {
              options = options.size(size.value()[size_i]);
              size_i++;
              return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
            },
            input_structure);
        return NestedTensor(std::move(res));
      }
    }

    throw std::runtime_error("Either size or scale_factor should be defined.");
}

namespace py = pybind11;

void add_functions(	
    pybind11::module m) {
  m.def(
      "my_interpolate",
      [](at::Tensor input,
         c10::optional<std::vector<std::vector<int64_t>>> size,
         c10::optional<THPArrayRef<double>> scale_factor,
         c10::optional<std::string> mode,
         c10::optional<bool> align_corners,
         c10::optional<bool> recompute_scale_factor) {
      std::cout << "010101" << std::endl;
        auto input_impl = get_nested_tensor_impl(input);
        auto input_data = input_impl->_data;
        if (scale_factor.has_value() && size.has_value()) {
          throw std::runtime_error(
              "only one of size or scale_factor should be defined");
        }

        if (size.has_value()) {
          return at::detail::make_tensor<NestedTensorImpl>(interpolate(
              input_data, size.value(), c10::nullopt, mode, align_corners));
        }

        if (scale_factor.has_value()) {
          return at::detail::make_tensor<NestedTensorImpl>(interpolate(
              input_data,
              c10::nullopt,
              scale_factor.value().extract<2>(),
              mode,
              align_corners));
        }

        throw "Either size or scale factor have to be passed.";
        // return at::ones({0});
      },
      py::arg("input"),
      py::arg("size") = nullptr,
      py::arg("scale_factor") = nullptr,
      py::arg("mode") = "nearest",
      py::arg("align_corners") = false,
      py::arg("recompute_scale_factor") = false);
}
} // namespace nested_tensor
} // namespace torch
