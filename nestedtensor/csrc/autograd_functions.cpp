#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {
struct NestedTensorFunction_conv2d
    : torch::autograd::Function<NestedTensorFunction_conv2d> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input_,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups) {
    auto autograd_input = map_nested_tensor(
        [](at::Tensor ti) {
          AutoGradMode autogradmode(true);
          auto alias = ti.alias();
          alias.requires_grad_();
          return alias;
        },
        input_);
    auto autograd_output = map_nested_tensor(
        [&](at::Tensor t) {
          AutoGradMode autogradmode(true);
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
    if (is_nested_tensor_impl(grad_output[0])) {
      grad = map_nested_tensor(
          [&](at::Tensor r, at::Tensor i, at::Tensor g) {
            // TODO: Might have to retain graph in many to one settings.
            std::vector<at::Tensor> result;
            if (bias) {
              result = torch::autograd::grad(
                  {r}, {i, weight, *bias}, {g}, c10::nullopt, false, true);
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
    } else {
      grad = map_nested_tensor(
          [&](at::Tensor r, at::Tensor i) {
            // TODO: Might have to retain graph in many to one settings.
            std::vector<at::Tensor> result;
            if (bias) {
              result = torch::autograd::grad(
                  {r},
                  {i, weight, *bias},
                  {grad_output[0]},
                  c10::nullopt,
                  false,
                  true);
            } else {
              result = torch::autograd::grad(
                  {r},
                  {i, weight},
                  {grad_output[0]},
                  c10::nullopt,
                  false,
                  true);
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
          autograd_input);
    }

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
    TORCH_CHECK(weight_, "asdf0");
    TORCH_CHECK(bias_, "asdf1");
    auto autograd_input = map_nested_tensor(
        [](at::Tensor ti) {
          AutoGradMode autogradmode(true);
          auto alias = ti.alias();
          alias.requires_grad_();
          return alias;
        },
        input_);
    at::Tensor weight;
    at::Tensor bias;
    {
      AutoGradMode autogradmode(true);
      weight = (*weight_).alias().requires_grad_();
      bias = (*bias_).alias().requires_grad_();
    }
    auto autograd_output = map_nested_tensor(
        [&](at::Tensor t) {
          AutoGradMode autogradmode(true);
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
    auto bias = ctx->saved_data["1"].toTensor();
    auto autograd_output = ctx->saved_data["2"].toTensor();
    auto autograd_input = ctx->saved_data["3"].toTensor();
    auto weight_grad = torch::zeros_like(weight);
    auto bias_grad = torch::zeros_like(bias);

    at::Tensor grad;
    TORCH_CHECK(grad_output.size() == 1, "not supported 0");
    if (is_nested_tensor_impl(grad_output[0])) {
      grad = map_nested_tensor(
          [&](at::Tensor r, at::Tensor i, at::Tensor g) {
            // TODO: Might have to retain graph in many to one settings.
            std::cout << "g11: " << g.sum() << std::endl;
            auto result = torch::autograd::grad(
                {r}, {i, weight, bias}, {g}); //, c10::nullopt, false, true);
            if (result[1].defined()) {
              std::cout << "HDHDHD 0" << std::endl;
              weight_grad.add_(result[1]);
            }
            if (result[2].defined()) {
              bias_grad.add_(result[2]);
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
    } else {
      grad = map_nested_tensor(
          [&](at::Tensor r, at::Tensor i) {
            // TODO: Might have to retain graph in many to one settings.
            std::cout << "grad_output[0]11: " << grad_output[0].sum()
                      << std::endl;
            // TODO: Print grad_output grad_fn and figure out whcih
            // input isn't used.
            auto result = torch::autograd::grad(
                {r},
                {i, weight, bias},
                {grad_output[0]},
                c10::nullopt,
                false,
                true);
            if (result[1].defined()) {
              std::cout << "HDHDHD 1" << std::endl;
              weight_grad.add_(result[1]);
            }
            if (result[2].defined()) {
              bias_grad.add_(result[2]);
            }
            if (result[0].defined()) {
              return result[0];
            }
            // TODO: NestedTensor doesn't support undefined devices yet.
            return torch::ones({1}).expand(i.sizes());
          },
          autograd_output,
          autograd_input);
    }

    at::Tensor undef;
    std::cout << "0 weight_grad: " << weight_grad.sum() << std::endl;
    std::cout << "0 bias_grad: " << bias_grad.sum() << std::endl;
    return {
        grad, weight_grad, bias_grad, undef, undef, undef, undef, undef, undef};
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

Tensor NestedTensor_relu(const Tensor& self) {
  return autograd_map_nested_tensor(
      [](at::Tensor tensor) { return at::relu(tensor); }, self);
}

struct NestedTensorFunction_relu_
    : public torch::autograd::Function<NestedTensorFunction_relu_> {
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor& self) {
    ctx->saved_data["0"] = self.clone();
    apply_nested_tensor([](at::Tensor& t) { at::relu_(t); }, self);
    ctx->mark_dirty({self});
    return self;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto result = ctx->saved_data["0"].toTensor();
    auto grad = grad_output[0];
    if (is_nested_tensor_impl(grad)) {
      return {map_nested_tensor(
          [](at::Tensor r, at::Tensor g) {
            TORCH_CHECK(
                !g.requires_grad(),
                "NestedTensor relu_ doesn't support double backward.");
            auto res = threshold_backward(g, r, 0);
            std::cout << "res0: " << res.sum() << std::endl;
            return res;
          },
          result,
          grad)};
    }
    TORCH_CHECK(
        !grad.requires_grad(),
        "NestedTensor relu_ doesn't support double backward.");
    return {map_nested_tensor(
        [&](at::Tensor r) {
          auto res = threshold_backward(grad, r, 0);
          std::cout << "res1: " << res.sum() << std::endl;
          return res;
        },
        result)};
  }
};

Tensor& NestedTensor_relu_(Tensor& self) {
  NestedTensorFunction_relu_::apply(self);
  return self;
}

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  // return autograd_map_nested_tensor(
  //     [&](const at::Tensor t) { return at::dropout(t, p, train); }, input);
  return input;
}

Tensor& NestedTensor_dropout_(Tensor& input, double p, bool train) {
  // apply_nested_tensor(
  //     [&](at::Tensor t) { return at::dropout_(t, p, train); }, input);
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

struct NestedTensorFunction_add_
    : public torch::autograd::Function<NestedTensorFunction_add_> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      Tensor& self,
      const Tensor& other,
      Scalar alpha) {
    if (is_nested_tensor_impl(other)) {
      apply_nested_tensor(
          [&](at::Tensor& s, Tensor o) { at::native::add_(s, o, alpha); },
          self,
          other);
    } else {
      apply_nested_tensor(
          [&](at::Tensor& s) { at::native::add_(s, other, alpha); }, self);
    }
    ctx->saved_data["0"] = alpha;
    ctx->mark_dirty({self});
    return self;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto alpha = ctx->saved_data["0"].toScalar();
    auto grad = grad_output[0];
    at::Tensor undef;
    //TODO: This is incorrect?
    return {grad,
      grad,
      undef};
//            map_nested_tensor(
//                [&](at::Tensor g) {
//                  TORCH_CHECK(
//                      !g.requires_grad(),
//                      "NestedTensor add_ doesn't support double backward.");
//                  auto res = g * alpha;
//                  std::cout << "3030: " << res.sum() << std::endl;
//                  return res;
//                },
//                grad),
//            undef};
  }
};

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  // std::cout << "NestedTensor_add_ NestedTensor_add_" << std::endl;
  NestedTensorFunction_add_::apply(self, other, alpha);
  return self;
  // if (is_nested_tensor_impl(other)) {
  //   apply_nested_tensor(
  //       [alpha](Tensor& self, Tensor& other) { self.add_(other, alpha); },
  //       self,
  //       other);
  //   return self;
  // }
  // apply_nested_tensor(
  //     [&other, alpha](at::Tensor& self) { return self.add_(other, alpha);
  //     }, self);
  // return self;
}

Tensor NestedTensor_sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  return NestedTensorFunction_sum::apply(self, dtype);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  m.impl_UNBOXED("conv2d", NestedTensor_conv2d);
  m.impl_UNBOXED("batch_norm", NestedTensor_batch_norm);
  m.impl_UNBOXED("max_pool2d", NestedTensor_max_pool2d);
  m.impl_UNBOXED("relu", NestedTensor_relu);
  m.impl_UNBOXED("relu_", NestedTensor_relu_);
  m.impl_UNBOXED("dropout", NestedTensor_dropout);
  m.impl_UNBOXED("dropout_", NestedTensor_dropout_);
  // m.impl_UNBOXED("add_.Tensor", NestedTensor_add_);
  m.impl_UNBOXED("sum", NestedTensor_sum);
}

}
