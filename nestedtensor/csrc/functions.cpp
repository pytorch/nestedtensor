#include <functions.h>
#include <utils/nested_node_functions.h>
#include <torch/extension.h>

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

// TODO: If size(0) is 1 and we squeeze should this turn into a Tensor?
// Squeeze doens't touch the underlying data and is effectively a meta-data
// operation
NestedTensor squeeze(
    NestedTensor input,
    c10::optional<int64_t> dim_,
    c10::optional<NestedTensor> out) {
  if (out) {
    return out->copy_(squeeze(input, dim_, c10::nullopt));
  }
  auto sizes = input.sizes();
  if (!dim_) {
    // TODO: First dimension is always ignored.
    // We could decide to return a Tensor if the 0th
    // dimension can be squeezed.
    NestedTensor result = input;
    for (size_t i = 0; i < sizes.size() - 1; i++) {
      size_t index = sizes.size() - i - 1;
      c10::optional<int64_t> s = sizes[index];
      if (s && ((*s) == 1)) {
        result = squeeze(result, index, c10::nullopt);
      }
    }
    return result;
  }
  int64_t dim = at::maybe_wrap_dim(*dim_, input.dim());
  TORCH_CHECK(dim > 0, "Cannot squeeze first dimension.");
  TORCH_CHECK(
      ((sizes[dim]) && ((*(sizes[dim])) == 1)),
      "Given dimension is either undefined or not a singleton.");
  TensorNode structure = input.get_structure();
  int64_t nested_dim = input.nested_dim();
  if (dim < nested_dim) {
    structure = _squeeze_nested_dim(structure, dim);
  } else {
    int64_t height = structure.height();
    structure =
        map([dim, height](
                at::Tensor tensor) { return tensor.squeeze(dim - height); },
            structure);
  }
  if (input.get_buffer()) {
    at::Tensor buffer = *input.get_buffer();
    return NestedTensor(std::move(buffer), std::move(structure));
  }
  return NestedTensor(std::move(structure));
}

NestedTensor relu(NestedTensor input, 
                  c10::optional<bool> inplace) {
  TensorNode input_structure = input.get_structure();
  TensorNode res = map([&](at::Tensor t){
      return torch::relu(t.unsqueeze(0)).squeeze(0);
  }, input_structure);

  return NestedTensor(std::move(res));
}

NestedTensor relu_out(NestedTensor& input) {
  input = relu(input, true);
  return input;
}

NestedTensor dropout(NestedTensor input, 
                     c10::optional<double> p, 
                     c10::optional<bool> training, 
                     c10::optional<bool> inplace) {
  TensorNode input_structure = input.get_structure();
  TensorNode res = map([&](at::Tensor t){
      return torch::dropout(t, p.value(), training.value());
  }, input_structure);

  return NestedTensor(std::move(res));
}

NestedTensor conv2d(const NestedTensor input, 
                    const at::Tensor weight, 
                    c10::optional<at::Tensor> bias, 
                    c10::optional<std::vector<int64_t>> stride,
                    c10::optional<std::vector<int64_t>> padding,
                    c10::optional<std::vector<int64_t>> dilation,
                    c10::optional<int64_t> groups) {
  TensorNode structure = input.get_structure();
  auto options = F::Conv2dFuncOptions().stride(stride.value())
                                       .padding(padding.value())
                                       .dilation(dilation.value())
                                       .groups(groups.value());
  if (bias.has_value()) {
      options = options.bias(bias.value());
  }

  TensorNode res = map([&, options](at::Tensor t){
      return F::conv2d(t.unsqueeze(0), weight, options).squeeze(0);
  }, structure);

  return NestedTensor(std::move(res));
}

NestedTensor max_pool2d(NestedTensor input,
                        at::IntArrayRef kernel_size,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        at::IntArrayRef dilation,
                        bool ceil_mode) {
  TensorNode structure = input.get_structure();
  F::MaxPool2dFuncOptions options = F::MaxPool2dFuncOptions(kernel_size).stride(stride)
                                                                        .padding(padding)
                                                                        .dilation(dilation)
                                                                        .ceil_mode(ceil_mode);

  TensorNode res = map([&, options](at::Tensor t){
      return F::max_pool2d(t.unsqueeze(0), options).squeeze(0);
  }, structure);

  return NestedTensor(std::move(res));
}

NestedTensor batch_norm(NestedTensor input,
                        const at::Tensor running_mean,
                        const at::Tensor running_var,
                        c10::optional<at::Tensor> weight,
                        c10::optional<at::Tensor> bias,
                        bool training, 
                        double momentum,
                        double eps) {
    TensorNode structure = input.get_structure();
    
    auto options = F::BatchNormFuncOptions().momentum(momentum)
                                            .eps(eps)
                                            .training(training);

    if (weight.has_value()) {
        options = options.weight(weight.value());
    }

    if (bias.has_value()) {
        options = options.bias(bias.value());
    }

    TensorNode res = map([&, options](at::Tensor t){
        return F::batch_norm(t.unsqueeze(0), running_mean, running_var, options).squeeze(0);
    }, structure);

    return NestedTensor(std::move(res));
}

NestedTensor cross_entropy(NestedTensor input,
                           NestedTensor target,
                           c10::optional<at::Tensor> weight,
                           c10::optional<bool> size_average, // TODO: use
                           c10::optional<int64_t> ignore_index,
                           c10::optional<bool> reduce, // TODO: use
                           c10::optional<std::string> reduction) {
  TensorNode input_structure = input.get_structure();
  TensorNode target_structure = target.get_structure();
  F::CrossEntropyFuncOptions::reduction_t redct;
  if (reduction.value() == "mean" || reduction.value() == "none") {
      redct = torch::kMean;
  } else if (reduction.value() == "sum") {
      redct = torch::kSum;
  } else {
      std::stringstream error;
      error << "Unexpected mode for reduction: " << reduction.value() << std::endl;
      throw error;
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

NestedTensor interpolate(NestedTensor input,
                         c10::optional<std::vector<int64_t>> size,
                         c10::optional<std::vector<double>> scale_factor,
                         c10::optional<std::string> mode,
                         c10::optional<bool> align_corners) {
                         //bool recompute_scale_factor) { // TODO: use
    TensorNode input_structure = input.get_structure();
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
        std::stringstream error;
        error << "Unexpected mode for interpolate: " << mode.value() << std::endl;
        throw error;
    }

    auto options = F::InterpolateFuncOptions().mode(int_mode);
    if (scale_factor.has_value()) {
        options.scale_factor() = scale_factor.value();
    }

    if (align_corners.has_value()) {
        options.align_corners() = align_corners.value();
    }

    if (size.has_value()) {
        if (size.value().size() == 2) {
            options.size() = size.value();
        } else {
            options.size() = std::vector<int64_t>({size.value()[0], size.value()[0]});
        }
    }

    TensorNode res = map(
        [&options, &size, &scale_factor](at::Tensor input_tensor) {
          // size or scale factor have to be defined
          if (!size.has_value() && !scale_factor.has_value()) {
            std::vector<int64_t> sizes;
            sizes.push_back(input_tensor.unsqueeze(0).size(2));
            sizes.push_back(input_tensor.unsqueeze(0).size(2));
            options.size() = sizes;
          }

          return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
        },
        input_structure);

    return NestedTensor(std::move(res));
}

}
}
