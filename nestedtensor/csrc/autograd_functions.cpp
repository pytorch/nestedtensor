#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

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
    at::Tensor undef;
    ctx->save_for_backward({weight ? *weight : undef,
                            bias ? *bias : undef,
                            autograd_output,
                            autograd_input});
    return map_nested_tensor(
        [](at::Tensor t) { return t.detach(); }, autograd_output);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto saved_data = ctx->get_saved_variables();

    c10::optional<at::Tensor> weight;
    c10::optional<at::Tensor> bias;
    if (saved_data[0].defined()) {
      weight = saved_data[0];
    }
    if (saved_data[1].defined()) {
      bias = saved_data[1];
    }
    auto autograd_output = saved_data[2];
    auto autograd_input = saved_data[3];
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
  if (is_packed(self) && self.dim() == 3 && other.dim() == 1) {
#ifdef TRACEPACKED
    std::cout << "calling packed add" << std::endl;
#endif
    auto self_structure = get_nested_tensor_structure(self);
    auto self_impl = get_nested_tensor_impl(self);
    return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
        (*self_structure.buffer())
            .reshape({-1, other.size(0)})
            .add(other)
            .reshape({-1}),
        self_impl->nested_size()));
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
  nt_impl(m, "batch_norm", NestedTensor_batch_norm);
  nt_impl(m, "max_pool2d", NestedTensor_max_pool2d);
  nt_impl(m, "sum", NestedTensor_sum);
  // nt_impl(m, "upsample_bilinear2d", NestedTensor_upsample_bilinear2d);
  nt_impl(m, "clone", NestedTensor_clone);
  nt_impl(m, "dropout", NestedTensor_dropout);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  nt_impl(m, "add.Tensor", NestedTensor_add);
  nt_impl(m, "add_.Tensor", NestedTensor_add_);
  nt_impl(m, "relu", NestedTensor_relu);
  nt_impl(m, "relu_", NestedTensor_relu_);
  nt_impl(m, "threshold_backward", NestedTensor_threshold_backward);
}

} // namespace at
