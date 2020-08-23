#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

namespace impl {
// def _grad_input_padding(grad_output, input_size, stride, padding,
// kernel_size, dilation=None):
//     if dilation is None:
//         # For backward compatibility
//         warnings.warn("_grad_input_padding 'dilation' argument not provided.
//         Default of 1 is used.") dilation = [1] * len(stride)
//
//     input_size = list(input_size)
//     k = grad_output.dim() - 2
//
//     if len(input_size) == k + 2:
//         input_size = input_size[-k:]
//     if len(input_size) != k:
//         raise ValueError("input_size must have {} elements (got {})"
//                          .format(k + 2, len(input_size)))
//
//     def dim_size(d):
//         return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
//         1
//                 + dilation[d] * (kernel_size[d] - 1))
//
//     min_sizes = [dim_size(d) for d in range(k)]
//     max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
//     for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
//         if size < min_size or size > max_size:
//             raise ValueError(
//                 ("requested an input grad size of {}, but valid sizes range "
//                  "from {} to {} (for a grad_output of {})").format(
//                      input_size, min_sizes, max_sizes,
//                      grad_output.size()[2:]))
//
//     return tuple(input_size[d] - min_sizes[d] for d in range(k))
//
//    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
//                                             padding, kernel_size, dilation)

//    kernel_size = [weight.shape[2]]
std::vector<int64_t> _grad_input_padding(
    at::Tensor grad_output,
    IntArrayRef input_size_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef kernel_size,
    IntArrayRef dilation) {
  std::cout << "grad_output.sizes(): " << grad_output.sizes() << std::endl;
  std::cout << "input_size_: " << input_size_ << std::endl;
  std::cout << "stride: " << stride << std::endl;
  std::cout << "padding: " << stride << std::endl;
  std::cout << "kernel_size: " << kernel_size << std::endl;
  std::cout << "dilation: " << dilation << std::endl;
  int64_t k = grad_output.dim() - 2;
  std::cout << "k: " << k << std::endl;

  std::vector<int64_t> input_size;

  if (input_size_.size() == k + 2) {
    for (int64_t i = 2; i < k + 2; i++) {
      input_size.push_back(input_size_[i]);
    }
    // input_size = input_size[-k:] // TODO
    // TORCH_CHECK(false, "NOT IMPLEMENTED");
  } else {
    input_size = input_size_.vec();
  }
  TORCH_CHECK(
      input_size.size() == k,
      "input_size must have ",
      k + 2,
      " elements (got ",
      input_size_.size(),
      ")");

  // def dim_size(d):
  //     return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 1
  //             + dilation[d] * (kernel_size[d] - 1))

  // min_sizes = [dim_size(d) for d in range(k)]
  // max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
  // for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
  //     if size < min_size or size > max_size:
  //         raise ValueError(
  //             ("requested an input grad size of {}, but valid sizes range "
  //              "from {} to {} (for a grad_output of {})").format(
  //                  input_size, min_sizes, max_sizes,
  //                  grad_output.size()[2:]))
  // return tuple(input_size[d] - min_sizes[d] for d in range(k))
  std::vector<int64_t> result_size;
  for (int64_t d = 0; d < k; d++) {
    int64_t min_size = ((grad_output.size(d + 2) - 1) * stride[d]) -
        (2 * padding[d]) + 1 + (dilation[d] * (kernel_size[d] - 1));
    int64_t max_size = min_size + stride[d] - 1;
    TORCH_CHECK(
        !(input_size[d] < min_size || input_size[d] > max_size),
        "input grad size outside of valid range. input_size[",
        d,
        "]: ",
        input_size[d],
        " min_size: ",
        min_size,
        " max_size: ",
        max_size);
    result_size.push_back(input_size[d] - min_size);
  }
  return result_size;
}
} // namespace impl

struct NestedTensorFunction_conv2d
    : torch::autograd::Function<NestedTensorFunction_conv2d> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input_,
      const Tensor& weight_,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups) {
    ctx->saved_data["4"] = stride.vec();
    ctx->saved_data["5"] = padding.vec();
    ctx->saved_data["6"] = groups;
    std::cout << "fw dilation: " << dilation << std::endl;
    ctx->saved_data["7"] = dilation.vec();
    auto autograd_input = map_nested_tensor(
        [](at::Tensor ti) {
          AutoGradMode autogradmode(true);
          auto alias = ti.alias();
          alias.requires_grad_();
          return alias;
        },
        input_);
    at::Tensor weight;
    {
      AutoGradMode autogradmode(true);
      weight = weight_.alias().requires_grad_();
    }
    auto autograd_output = map_nested_tensor(
        [&](at::Tensor t) {
          AutoGradMode autogradmode(true);
          return at::conv2d(
                     t.unsqueeze(0),
                     weight,
                     bias,
                     stride,
                     padding,
                     dilation,
                     groups)
              .squeeze(0);
        },
        autograd_input);
    ctx->saved_data["0"] = weight;
    ctx->saved_data["1"] = bias;
    ctx->saved_data["2"] = autograd_output;
    ctx->saved_data["3"] = autograd_input;

    return map_nested_tensor(
        [](at::Tensor t) { return t.detach(); }, autograd_output);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto weight = ctx->saved_data["0"].toTensor();
    auto bias = ctx->saved_data["1"].toOptional<at::Tensor>();
    auto autograd_output = ctx->saved_data["2"].toTensor();
    auto autograd_input = ctx->saved_data["3"].toTensor();

    auto stride = ctx->saved_data["4"].toIntList().vec();
    auto padding = ctx->saved_data["5"].toIntList().vec();
    auto groups = ctx->saved_data["6"].toInt();
    auto dilation = ctx->saved_data["7"].toIntList().vec();
    std::cout << "bw dilation: " << dilation << std::endl;

    auto weight_grad = torch::zeros_like(weight);
    c10::optional<at::Tensor> bias_grad;
    if (bias) {
      bias_grad = torch::zeros_like(*bias);
    }
    bool grad_undefined = false;
    bool weight_grad_undefined = false;
    bool bias_grad_undefined = false;

    at::Tensor grad;
    TORCH_CHECK(grad_output.size() == 1, "not supported 0");
    grad = map_nested_tensor(
        [&](at::Tensor r, at::Tensor i, at::Tensor g) {
          // TODO: Might have to retain graph in many to one settings.
          std::vector<at::Tensor> result;
          if (bias) {
            // >>> input = torch.randn(1,1,3,3, requires_grad=True)
            // >>> weight = torch.randn(1,1,1,2, requires_grad=True)
            // >>> output = F.conv2d(input, weight)
            // >>> grad_output = torch.randn(output.shape)
            // >>> grad_input = torch.autograd.grad(output, input, grad_output)
            // >>> F.grad.conv2d_input(input.shape, weight, grad_output)
            // at::Tensor conv_transpose2d(
            //     const Tensor& input, const Tensor& weight, const Tensor&
            //     bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef
            //     output_padding, int64_t groups, IntArrayRef dilation) {
            std::cout << "has bias with sizes " << (*bias).sizes() << std::endl;
            std::cout << "i.sizes(): " << i.sizes() << std::endl;
            auto i_ = i.unsqueeze(0);
            auto g_ = g.unsqueeze(0);

            std::vector<int64_t> kernel_size{weight.size(2), weight.size(3)};
            auto grad_input_padding = impl::_grad_input_padding(
                g_,
                i_.sizes(),
                IntArrayRef(stride),
                IntArrayRef(padding),
                IntArrayRef(kernel_size),
                IntArrayRef(dilation));
            result.push_back(at::conv_transpose2d(
                g_,
                weight,
                *bias,
                IntArrayRef(stride),
                IntArrayRef(padding),
                IntArrayRef(grad_input_padding),
                groups,
                IntArrayRef(dilation)).squeeze(0));
            // result = torch::autograd::grad(
            //     {r}, {i, weight, *bias}, {g}, c10::nullopt, false, true);
          } else {
            result = torch::autograd::grad(
                {r}, {i, weight}, {g}, c10::nullopt, false, true);
          }
          if (!result[1].defined()) {
            weight_grad_undefined = true;
          } else {
            weight_grad.add_(result[1]);
          }
          if (result[2].defined() && bias) {
            (*bias_grad).add_(result[2]);
          } else {
            bias_grad_undefined = true;
          }
          if (!result[0].defined()) {
            grad_undefined = true;
            // TODO: NestedTensor doesn't support undefined devices yet.
            return torch::ones({0});
          }
          return result[0];
        },
        autograd_output,
        autograd_input,
        grad_output[0]);

    at::Tensor undef;
    return {grad_undefined ? undef : grad,
            weight_grad_undefined ? undef : weight_grad,
            bias_grad_undefined || !bias ? undef : *bias_grad,
            undef,
            undef,
            undef,
            undef,
            undef};
  }
};

// TODO: Cover all the cases!
struct NestedTensorFunction_batch_norm
    : torch::autograd::Function<NestedTensorFunction_batch_norm> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input_,
      const c10::optional<Tensor>& weight_,
      const c10::optional<Tensor>& bias_,
      const c10::optional<Tensor>& running_mean,
      const c10::optional<Tensor>& running_var,
      bool training,
      double momentum,
      double eps,
      bool cudnn_enabled) {
    // TORCH_CHECK(weight_, "asdf0");
    // TORCH_CHECK(bias_, "asdf1");
    c10::optional<at::Tensor> weight;
    c10::optional<at::Tensor> bias;
    auto autograd_input = map_nested_tensor(
        [](at::Tensor ti) {
          AutoGradMode autogradmode(true);
          auto alias = ti.alias();
          alias.requires_grad_();
          return alias;
        },
        input_);
    {
      AutoGradMode autogradmode(true);
      if (weight_) {
        weight = (*weight_).alias().detach().requires_grad_();
      }
      if (bias_) {
        bias = (*bias_).alias().detach().requires_grad_();
      }
    }
    auto autograd_output = map_nested_tensor(
        [&](at::Tensor t) {
          AutoGradMode autogradmode(true);
          return at::native::batch_norm(
                     t.unsqueeze(0),
                     *weight,
                     *bias,
                     *running_mean,
                     *running_var,
                     training,
                     momentum,
                     eps,
                     cudnn_enabled)
              .squeeze(0);
        },
        autograd_input);
    ctx->saved_data["0"] = weight;
    ctx->saved_data["1"] = bias;
    ctx->saved_data["2"] = autograd_output;
    ctx->saved_data["3"] = autograd_input;
    return map_nested_tensor(
        [](at::Tensor t) { return t.detach(); }, autograd_output);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto weight = ctx->saved_data["0"].toOptional<at::Tensor>();
    auto bias = ctx->saved_data["1"].toOptional<at::Tensor>();
    auto autograd_output = ctx->saved_data["2"].toTensor();
    auto autograd_input = ctx->saved_data["3"].toTensor();
    c10::optional<at::Tensor> weight_grad;
    if (weight) {
      weight_grad = torch::zeros_like(*weight);
    }
    c10::optional<at::Tensor> bias_grad;
    if (bias) {
      bias_grad = torch::zeros_like(*bias);
    }

    at::Tensor grad;
    TORCH_CHECK(grad_output.size() == 1, "not supported 0");
    grad = map_nested_tensor(
        [&](at::Tensor r, at::Tensor i, at::Tensor g) {
          // TODO: Might have to retain graph in many to one settings.
          std::vector<at::Tensor> inputs;
          inputs.push_back(i);
          if (weight) {
            inputs.push_back(*weight);
          }
          if (bias) {
            inputs.push_back(*bias);
          }
          auto result = torch::autograd::grad(
              {r}, inputs, {g}, c10::nullopt, false, true);
          if (result[1].defined()) {
            (*weight_grad).add_(result[1]);
          }
          if (result[2].defined()) {
            (*bias_grad).add_(result[2]);
          }
          if (result[0].defined()) {
            return result[0];
          }
          // TODO: NestedTensor doesn't support undefined devices yet.
          return torch::ones({1}).expand(i.sizes());
        },
        autograd_output,
        autograd_input,
        grad_output[0]);

    at::Tensor undef;
    return {grad,
            weight_grad ? *weight_grad : undef,
            bias_grad ? *bias_grad : undef,
            undef,
            undef,
            undef,
            undef,
            undef,
            undef};
  }
};

Tensor NestedTensor_batch_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  return NestedTensorFunction_batch_norm::apply(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      cudnn_enabled);
}

Tensor NestedTensor_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  return NestedTensorFunction_conv2d::apply(
      input, weight, bias, stride, padding, dilation, groups);
}

Tensor NestedTensor_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  return autograd_map_nested_tensor(
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
      self);
}

// Registered below autograd
Tensor NestedTensor_relu(const Tensor& self) {
  auto impl = get_nested_tensor_impl(self);
  auto structure = get_nested_tensor_structure(self);
  if (structure.buffer()) {
#ifdef TRACEPACKED
    std::cout << "calling packed relu" << std::endl;
#endif
    return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
        at::relu(*structure.buffer()), impl->nested_size()));
  }
  return map_nested_tensor(
      [](at::Tensor tensor) { return at::relu(tensor); }, self);
}

// Registered below autograd
Tensor& NestedTensor_relu_(Tensor& self) {
  apply_nested_tensor([](at::Tensor& tensor) { at::relu_(tensor); }, self);
  return self;
}

// Registered below autograd
Tensor NestedTensor_threshold_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar threshold) {
  return map_nested_tensor(
      [&](at::Tensor g, at::Tensor s) {
        return threshold_backward(g, s, threshold);
      },
      grad,
      self);
}

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  return autograd_map_nested_tensor(
      [&](const at::Tensor t) { return at::dropout(t, p, train); }, input);
}

Tensor& NestedTensor_dropout_(Tensor& input, double p, bool train) {
  throw std::runtime_error("dropout_ is not implemented");
  return input;
}

struct NestedTensorFunction_sum
    : public torch::autograd::Function<NestedTensorFunction_sum> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input_,
      c10::optional<ScalarType> dtype) {
    auto input = map_nested_tensor(
        [](Tensor t) {
          auto alias = t.alias();
          alias.requires_grad_();
          return alias;
        },
        input_);
    auto tensors = flatten(map(
        [&dtype](at::Tensor tensor) {
          AutoGradMode autogradmode(true);
          return at::sum(tensor, dtype);
        },
        get_nested_tensor_structure(input)));
    Tensor result;
    {
      AutoGradMode autogradmode(true);
      if (tensors.size() == 0) {
        if (dtype) {
          return at::ones({0}, *dtype);
        }
        return at::ones({0});
      }
      auto all_tensor = at::stack(tensors);
      result = at::sum(all_tensor, dtype);
    }
    ctx->save_for_backward({result, input});
    return result.alias();
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    auto saved = ctx->get_saved_variables();
    at::Tensor result = saved[0];
    at::Tensor input = saved[1];
    at::Tensor grad_output = grad_output_[0];
    TORCH_CHECK(
        !grad_output.requires_grad(),
        "NestedTensor sum doesn't support double backward.");
    Tensor undef;
    // TODO:
    // Flatten constituents and call grad on all of the variable lists at once
    //
    at::Tensor tensor = map_nested_tensor(
        [&](Tensor i) {
          // return grad_output.expand(i.sizes());
          return torch::autograd::grad({result}, {i}, {grad_output}, true)[0];
        },
        input);
    return {tensor, undef};
  }
};

Tensor NestedTensor_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  if (is_nested_tensor_impl(self, other)) {
    return map_nested_tensor(
        [&](at::Tensor s, at::Tensor o) { return at::add(s, o, alpha); },
        self,
        other);
  }
  if (is_nested_tensor_impl(other)) {
    return map_nested_tensor(
        [&](at::Tensor o) { return at::add(self, o, alpha); }, other);
  }
  return map_nested_tensor(
      [&](at::Tensor s) { return at::add(s, other, alpha); }, self);
}

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  if (is_nested_tensor_impl(self, other)) {
    apply_nested_tensor(
        [&](at::Tensor& s, at::Tensor o) { s.add_(o, alpha); }, self, other);
    return self;
  }
  if (is_packed(self) && self.dim() == 3 && other.dim() == 1) {
#ifdef TRACEPACKED
    std::cout << "calling packed add_" << std::endl;
#endif
    auto self_structure = get_nested_tensor_structure(self);
    (*self_structure.buffer()).reshape({-1, other.size(0)}).add_(other);
    return self;
  }
  apply_nested_tensor([&](at::Tensor& s) { s.add_(other, alpha); }, self);
  return self;
}

Tensor NestedTensor_sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  return NestedTensorFunction_sum::apply(self, dtype);
}

Tensor NestedTensor_upsample_bilinear2d(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  return autograd_map_nested_tensor(
      [&](at::Tensor t) {
        return at::upsample_bilinear2d(
                   t.unsqueeze(0),
                   output_size,
                   align_corners,
                   scales_h,
                   scales_w)
            .squeeze(0);
      },
      input);
}

Tensor NestedTensor_clone(
    const Tensor& src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return autograd_map_nested_tensor(
      [&optional_memory_format](Tensor a) {
        return at::clone(a, optional_memory_format);
      },
      src);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
  nt_impl(m, "batch_norm", NestedTensor_batch_norm);
  nt_impl(m, "max_pool2d", NestedTensor_max_pool2d);
  nt_impl(m, "sum", NestedTensor_sum);
  // nt_impl(m, "upsample_bilinear2d", NestedTensor_upsample_bilinear2d);
  nt_impl(m, "clone", NestedTensor_clone);
  nt_impl(m, "dropout", NestedTensor_dropout);
  nt_impl(m, "dropout_", NestedTensor_dropout_);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  nt_impl(m, "add.Tensor", NestedTensor_add);
  nt_impl(m, "add_.Tensor", NestedTensor_add_);
  nt_impl(m, "relu", NestedTensor_relu);
  nt_impl(m, "relu_", NestedTensor_relu_);
  nt_impl(m, "threshold_backward", NestedTensor_threshold_backward);
}

} // namespace at
