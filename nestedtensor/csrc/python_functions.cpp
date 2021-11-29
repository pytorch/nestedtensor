#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_args.h>
#include <nestedtensor/csrc/python_functions.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace torch {
namespace nested_tensor {

at::Tensor cross_entropy(
    at::Tensor input,
    at::Tensor target,
    c10::optional<at::Tensor>& weight,
    c10::optional<bool>& size_average, // TODO: use
    c10::optional<int64_t>& ignore_index,
    c10::optional<bool>& reduce, // TODO: use
    c10::optional<std::string>& reduction,
    c10::optional<double> label_smoothing) {
  F::CrossEntropyFuncOptions::reduction_t redct;
  if (reduction.value() == "mean" || reduction.value() == "none") {
    redct = torch::kMean;
  } else if (reduction.value() == "sum") {
    redct = torch::kSum;
  } else {
    throw std::runtime_error(
        "Unexpected mode for reduction: " + reduction.value());
  }

  auto options = F::CrossEntropyFuncOptions().reduction(redct);
  if (ignore_index.has_value()) {
    options = options.ignore_index(ignore_index.value());
  }
  if (label_smoothing.has_value()) {
    options = options.label_smoothing(label_smoothing.value());
  }

  return map_nested_tensor(
      [&, options](at::Tensor input_tensor, at::Tensor target_tensor) {
        return F::cross_entropy(
                   input_tensor.unsqueeze(0),
                   target_tensor.unsqueeze(0),
                   options)
            .squeeze(0);
      },
      input,
      target);
}

at::Tensor interpolate(
    at::Tensor input,
    c10::optional<std::vector<std::vector<int64_t>>> size,
    c10::optional<at::ArrayRef<double>> scale_factor,
    c10::optional<std::string> mode,
    c10::optional<bool> align_corners) {
  F::InterpolateFuncOptions::mode_t int_mode;
  if (mode.value() == "nearest" || mode.value() == "none") {
    int_mode = torch::kNearest;
  } else if (mode.value() == "trilinear") {
    int_mode = torch::kTrilinear;
  } else if (mode.value() == "linear") {
    int_mode = torch::kLinear;
  } else if (mode.value() == "bicubic") {
    int_mode = torch::kBicubic;
  } else if (mode.value() == "area") {
    int_mode = torch::kArea;
  } else if (mode.value() == "bilinear") {
    int_mode = torch::kBilinear;
  } else {
    throw std::runtime_error(
        "Unexpected mode for interpolate: " + mode.value());
  }

  auto options = F::InterpolateFuncOptions().mode(int_mode);
  if (align_corners.has_value()) {
    options.align_corners() = align_corners.value();
  }

  // Either scale factor or size can be passed
  if (scale_factor.has_value()) {
    options = options.scale_factor(scale_factor.value().vec());
    return map_nested_tensor(
        [&options](at::Tensor input_tensor) {
          return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
        },
        input);
  }

  // Get input leaves count
  auto leaves_count = reduce_nested_tensor(
      [](at::Tensor leaf, int64_t input) { return input + 1; }, 0, input);

  if (size.has_value()) {
    // There can be either 1 size for all tensor or an individual size value per
    // tensor
    if (size.value().size() != 1 && size.value().size() != leaves_count) {
      throw std::runtime_error(
          "Interpolate has to take either 1 size tuple or same amount as leaves in Nested Tensor.");
    }

    if (size.value().size() == 1) {
      return map_nested_tensor(
          [&options, &size](at::Tensor input_tensor) {
            options = options.size(size.value()[0]);
            return F::interpolate(input_tensor.unsqueeze(0), options)
                .squeeze(0);
          },
          input);
    } else {
      int size_i = 0;
      return map_nested_tensor(
          [&options, &size_i, &size](at::Tensor input_tensor) {
            options = options.size(size.value()[size_i]);
            size_i++;
            return F::interpolate(input_tensor.unsqueeze(0), options)
                .squeeze(0);
          },
          input);
    }
  }

  throw std::runtime_error("Either size or scale_factor should be defined.");
}

namespace py = pybind11;

void add_functions(pybind11::module m) {
  m.def(
      "interpolate",
      [](at::Tensor input,
         c10::optional<std::vector<std::vector<int64_t>>> size,
         c10::optional<THPArrayRef<double>> scale_factor,
         c10::optional<std::string> mode,
         c10::optional<bool> align_corners,
         c10::optional<bool> recompute_scale_factor,
         bool antialias) {
        if (scale_factor.has_value() && size.has_value()) {
          throw std::runtime_error(
              "only one of size or scale_factor should be defined");
        }
        if (antialias) {
          throw std::runtime_error("Antialias is not yet supported");
        }

        if (size.has_value()) {
          return interpolate(
              input, size.value(), c10::nullopt, mode, align_corners);
        }

        if (scale_factor.has_value()) {
          return interpolate(
              input,
              c10::nullopt,
              scale_factor.value().extract<2>(),
              mode,
              align_corners);
        }

        throw std::runtime_error(
            "Either size or scale factor have to be passed.");
      },
      py::arg("input"),
      py::arg("size") = nullptr,
      py::arg("scale_factor") = nullptr,
      py::arg("mode") = "nearest",
      py::arg("align_corners") = false,
      py::arg("recompute_scale_factor") = false,
      py::arg("antialias") = false);

  m.def(
      "interpolate",
      [](at::Tensor input,
         c10::optional<std::vector<int64_t>> size,
         c10::optional<THPArrayRef<double>> scale_factor,
         c10::optional<std::string> mode,
         c10::optional<bool> align_corners,
         c10::optional<bool> recompute_scale_factor,
         bool antialias) {
        if (scale_factor.has_value() && size.has_value()) {
          throw std::runtime_error(
              "only one of size or scale_factor should be defined");
        }
        if (antialias) {
          throw std::runtime_error("Antialias is not yet supported");
        }

        if (size.has_value()) {
          std::vector<std::vector<int64_t>> sizes{size.value()};
          return interpolate(input, sizes, c10::nullopt, mode, align_corners);
        }

        if (scale_factor.has_value()) {
          return interpolate(
              input,
              c10::nullopt,
              scale_factor.value().extract<2>(),
              mode,
              align_corners);
        }

        throw std::runtime_error(
            "Either size or scale factor have to be passed.");
      },
      py::arg("input"),
      py::arg("size") = nullptr,
      py::arg("scale_factor") = nullptr,
      py::arg("mode") = "nearest",
      py::arg("align_corners") = false,
      py::arg("recompute_scale_factor") = false,
      py::arg("antialias") = false);

  m.def(
      "interpolate",
      [](at::Tensor input,
         c10::optional<int64_t> size,
         c10::optional<THPArrayRef<double>> scale_factor,
         c10::optional<std::string> mode,
         c10::optional<bool> align_corners,
         c10::optional<bool> recompute_scale_factor,
         bool antialias) {
        if (scale_factor.has_value() && size.has_value()) {
          throw std::runtime_error(
              "only one of size or scale_factor should be defined");
        }
        if (antialias) {
          throw std::runtime_error("Antialias is not yet supported");
        }

        if (size.has_value()) {
          std::vector<std::vector<int64_t>> sizes{
              std::vector<int64_t>{size.value(), size.value()}};

          return interpolate(input, sizes, c10::nullopt, mode, align_corners);
        }

        if (scale_factor.has_value()) {
          return interpolate(
              input,
              c10::nullopt,
              scale_factor.value().extract<2>(),
              mode,
              align_corners);
        }

        throw std::runtime_error(
            "Either size or scale factor have to be passed.");
      },
      py::arg("input"),
      py::arg("size") = nullptr,
      py::arg("scale_factor") = nullptr,
      py::arg("mode") = "nearest",
      py::arg("align_corners") = false,
      py::arg("recompute_scale_factor") = false,
      py::arg("antialias") = false);

  m.def(
      "cross_entropy",
      [](at::Tensor input,
         at::Tensor target,
         c10::optional<at::Tensor> weight,
         c10::optional<bool> size_average, // TODO: use
         c10::optional<int64_t> ignore_index,
         c10::optional<bool> reduce, // TODO: use
         c10::optional<std::string> reduction,
         c10::optional<double> label_smoothing) {
        return cross_entropy(
            input,
            target,
            weight,
            size_average,
            ignore_index,
            reduce,
            reduction,
            label_smoothing);
      },
      py::arg("input"),
      py::arg("target"),
      py::arg("weight") = nullptr,
      py::arg("size_average") = true,
      py::arg("ignore_index") = -100,
      py::arg("reduce") = true,
      py::arg("reduction") = "mean",
      py::arg("label_smoothing") = 0.0);
}
} // namespace nested_tensor
} // namespace torch
