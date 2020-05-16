#include <nestedtensor/csrc/functions.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace torch {
namespace nested_tensor {

inline TensorNode _squeeze_nested_dim(TensorNode structure, int64_t dim) {
  if (dim == 0) {
    return structure.children(0);
  }
  return TensorNode(_squeeze_nested_dim(structure, dim - 1));
}

NestedTensor cross_entropy(NestedTensor& input,
                           NestedTensor& target,
                           c10::optional<at::Tensor>& weight,
                           c10::optional<bool>& size_average, // TODO: use
                           c10::optional<int64_t>& ignore_index,
                           c10::optional<bool>& reduce, // TODO: use
                           c10::optional<std::string>& reduction) {
  TensorNode input_structure = input.get_structure();
  TensorNode target_structure = target.get_structure();
  F::CrossEntropyFuncOptions::reduction_t redct;
  if (reduction.value() == "mean" || reduction.value() == "none") {
      redct = torch::kMean;
  } else if (reduction.value() == "sum") {
      redct = torch::kSum;
  } else {
      throw std::runtime_error("Unexpected mode for reduction: " + reduction.value());
  }

  auto options = F::CrossEntropyFuncOptions().reduction(redct);
  if (ignore_index.has_value()) {
      options = options.ignore_index(ignore_index.value());
  }

  TensorNode res = map([&, options] (at::Tensor input_tensor, at::Tensor target_tensor){
      return F::cross_entropy(input_tensor.unsqueeze(0), target_tensor.unsqueeze(0), options).squeeze(0);
  }, input_structure, target_structure);

  return NestedTensor(std::move(res));
}

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
}
}

namespace at {

Tensor NestedTensor_batch_norm(
    const Tensor& input, const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  auto input_impl = get_nested_tensor_impl(input);
  auto input_data = input_impl->_data;
  auto structure = input_data.get_structure();
  auto res = map(
      [&](at::Tensor t) {
        return at::batch_norm(
                   t.unsqueeze(0),
                   weight,
                   bias,
                   running_mean,
                   running_var,
                   training,
                   momentum,
                   eps,
                   cudnn_enabled)
            .squeeze(0);
      },
      structure);
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)));
}

Tensor NestedTensor_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto input_impl = get_nested_tensor_impl(input);
  auto input_data = input_impl->_data;
  auto structure = input_data.get_structure();

  auto res = map(
      [&weight, &bias, &stride, &padding, &dilation, groups](at::Tensor t) {
        return at::convolution(
                   t.unsqueeze(0),
                   weight,
                   bias,
                   stride,
                   padding,
                   dilation,
                   false,
                   {{0, 0}},
                   groups)
            .squeeze(0);
      },
      structure);

  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)));
}

Tensor NestedTensor_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  auto self_impl = get_nested_tensor_impl(self);
  auto self_data = self_impl->_data;
  auto structure = self_data.get_structure();
  auto res = map(
      [&](at::Tensor t) {
        return at::max_pool2d(
                   t.unsqueeze(0),
                   kernel_size,
                   stride,
                   padding,
                   dilation,
                   ceil_mode)
            .squeeze(0);
      },
      structure);

  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)));
}

Tensor NestedTensor_relu(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto self_data = self_impl->_data;
  if (self_data.is_contiguous()) {
    auto res = torch::nested_tensor::NestedTensor(
        at::relu(*self_data.get_buffer()), self_data.nested_size());
    return at::detail::make_tensor<NestedTensorImpl>(std::move(res));
  }
  auto structure = self_data.get_structure();
  auto res =
      map([&](at::Tensor t) { return at::relu(t.unsqueeze(0)).squeeze(0); },
          structure);
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)));
}

Tensor & NestedTensor_relu_(Tensor & self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto self_data = self_impl->_data;
  if (self_data.is_contiguous()) {
    at::relu_(*self_data.get_buffer());
    return self;
  }
  auto structure = self_data.get_structure();
  apply([](at::Tensor& t) { at::relu_(t); }, structure);
  return self;
}

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  auto self_impl = get_nested_tensor_impl(input);
  auto self_data = self_impl->_data;
  auto structure = self_data.get_structure();
  auto res =
      map([&](const at::Tensor t) { return at::dropout(t, p, train); }, structure);
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)));
}

Tensor& NestedTensor_dropout_(Tensor& input, double p, bool train) {
  auto self_impl = get_nested_tensor_impl(input);
  auto self_data = self_impl->_data;
  auto structure = self_data.get_structure();
  apply([&](at::Tensor& t) { return at::dropout_(t, p, train); }, structure);
  return input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  m.impl_UNBOXED("conv2d", NestedTensor_conv2d);
  m.impl_UNBOXED("batch_norm", NestedTensor_batch_norm);
  m.impl_UNBOXED("max_pool2d", NestedTensor_max_pool2d);
  m.impl_UNBOXED("relu", NestedTensor_relu);
  m.impl_UNBOXED("relu_", NestedTensor_relu_);
  m.impl_UNBOXED("dropout", NestedTensor_dropout);
  m.impl_UNBOXED("dropout_", NestedTensor_dropout_);
}
}
