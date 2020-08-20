#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

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

Tensor NestedTensor_embedding(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  if (is_nested_tensor_impl(weight)) {
    // TODO: Needs test coverage
    return map_nested_tensor(
        [&](at::Tensor w, at::Tensor i) {
          return at::embedding(w, i, padding_idx, scale_grad_by_freq, sparse);
        },
        weight,
        indices);
  }
  return map_nested_tensor(
      [&](at::Tensor i) {
        return at::embedding(
            weight, i, padding_idx, scale_grad_by_freq, sparse);
      },
      indices);
}

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
            auto result = torch::autograd::grad(
                {r}, {i, weight, bias}, {g}, c10::nullopt, false, true);
            std::cout << "g11: " << g.sum() << std::endl;
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
            auto result = torch::autograd::grad(
                {r},
                {i, weight, bias},
                {grad_output[0]},
                c10::nullopt,
                false,
                true);
            std::cout << "grad_output[0]11: " << grad_output[0].sum()
                      << std::endl;
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

Tensor NestedTensor_reshape(const Tensor& self, IntArrayRef size) {
  auto self_data = get_nested_tensor_impl(self);
  TORCH_CHECK(
      int64_t(size.size()) > self_data->nested_dim(),
      "Reshape cannot be exclusive to nested dimensions.");
  for (int64_t i = 0; i < self_data->nested_dim(); i++) {
    if (size[i] >= 0) {
      throw std::runtime_error(
          "Cannot reshape explicitly along irregular dimension " +
          std::to_string(size[i]) + ". Please use -1 as a placeholder.");
    }
  }
  int64_t nested_dim = self_data->nested_dim();
  std::vector<int64_t> target_shape;
  for (int64_t i = nested_dim; i < int64_t(size.size()); i++) {
    target_shape.push_back(size[i]);
  }
  return map_nested_tensor(
      [target_shape](const at::Tensor t) {
        return at::reshape(t, IntArrayRef(target_shape));
      },
      self);
}

Tensor NestedTensor_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto self_data = get_nested_tensor_impl(self);
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }
  int64_t nested_dim = self_data->nested_dim();
  TORCH_CHECK(
      dim0 >= nested_dim && dim1 >= nested_dim,
      "Transposition of nested dimensions is not implemented yet.");
  return map_nested_tensor(
      [dim0, dim1, nested_dim](const at::Tensor t) {
        return at::transpose(t, dim0 - nested_dim, dim1 - nested_dim);
      },
      self);
}

Tensor NestedTensor_softmax(
    const Tensor& input,
    const int64_t dim_,
    c10::optional<ScalarType> dtype) {
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  auto input_data = get_nested_tensor_impl(input);
  int64_t nested_dim = input_data->nested_dim();
  TORCH_CHECK(
      dim >= nested_dim,
      "Cannot apply softmax across nested dimensions ",
      std::to_string(dim));
  return map_nested_tensor(
      [dim, nested_dim, dtype](const at::Tensor t) {
        return at::softmax(t, dim - nested_dim, dtype);
      },
      input);
}

Tensor NestedTensor_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  TORCH_CHECK(
      normalized_shape.size() == 1,
      "Currently only singleton tuples of integers supported for layer_norm.");
  auto input_data = get_nested_tensor_impl(input);
  TORCH_CHECK(
      input_data->opt_sizes()[input.dim() - 1],
      "Cannot normalize across irregular dimension ",
      std::to_string(input.dim() - 1));
  return map_nested_tensor(
      [normalized_shape, &weight, &bias, eps](const at::Tensor t) {
        return at::layer_norm(t, normalized_shape, weight, bias, eps, true);
      },
      input);
}

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
    return {grad,
            map_nested_tensor(
                [&](at::Tensor g) {
                  TORCH_CHECK(
                      !g.requires_grad(),
                      "NestedTensor add_ doesn't support double backward.");
                  auto res = g * alpha;
                  std::cout << "3030: " << res.sum() << std::endl;
                  return res;
                },
                grad),
            undef};
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

Tensor NestedTensor_all(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  if (self.numel() == 0) {
    // XXX: self.options doesn't work here because
    // we don't want a Tensor backed by a NestedTensor
    Tensor result = at::empty({0}, at::kBool); //, self.options());
    result.fill_(1);
    return result;
  }
  auto map_all = flatten(
      map([](at::Tensor tensor) { return tensor.all(); },
          self_impl->get_structure()));
  at::Tensor gathered = at::empty(
      {static_cast<int64_t>(map_all.size())}, at::kBool); //, self.options());
  for (size_t i = 0; i < map_all.size(); i++) {
    gathered[i] = map_all[i];
  }
  return gathered.all();
}

Tensor NestedTensor_any(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  if (self.numel() == 0) {
    // XXX: self.options doesn't work here because
    // we don't want a Tensor backed by a NestedTensor
    Tensor result = at::empty({0}, at::kBool); //, self.options());
    result.fill_(1);
    return result;
  }
  auto map_any = flatten(
      map([](at::Tensor tensor) { return tensor.any(); },
          self_impl->get_structure()));
  at::Tensor gathered = at::empty(
      {static_cast<int64_t>(map_any.size())}, at::kBool); //, self.options());
  for (size_t i = 0; i < map_any.size(); i++) {
    gathered[i] = map_any[i];
  }
  return gathered.any();
}

Tensor NestedTensor__log_softmax(
    const Tensor& self,
    const int64_t dim_,
    const bool half_to_float) {
  return map_nested_tensor(
      [&](Tensor a) { return at::_log_softmax(a, dim_, half_to_float); }, self);
}

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(other)) {
    return autograd_map_nested_tensor(
        [](Tensor tensor, Tensor other) { return at::matmul(tensor, other); },
        self,
        other);
  }
  return autograd_map_nested_tensor(
      [&other](Tensor tensor) { return at::matmul(tensor, other); }, self);
}

Tensor& NestedTensor_matmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  apply_nested_tensor(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        return at::matmul_out(result, tensor, other);
      },
      result,
      self,
      other);
  return result;
}

Tensor NestedTensor_pin_memory(const Tensor& self) {
  return map_nested_tensor(
      [](Tensor tensor) { return at::native::pin_memory(tensor); }, self);
}

Tensor NestedTensor_flatten(
    const Tensor& self,
    int64_t start_dim,
    int64_t end_dim) {
  auto self_data = get_nested_tensor_impl(self);
  start_dim = maybe_wrap_dim(start_dim, self.dim());
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  int64_t nested_dim = self_data->nested_dim();
  TORCH_CHECK(
      start_dim >= nested_dim, "Cannot flatten nested dimension ", start_dim);
  TORCH_CHECK(
      end_dim >= nested_dim, "Cannot flatten nested dimension ", end_dim);
  return map_nested_tensor(
      [start_dim, end_dim, nested_dim](at::Tensor tensor) {
        return at::flatten(
            tensor, start_dim - nested_dim, end_dim - nested_dim);
      },
      self);
}

std::vector<Tensor> get_stack_inputs(TensorList tensors, int64_t dim) {
  std::vector<Tensor> inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return inputs;
}

Tensor& NestedTensor_stack_out(
    Tensor& result,
    TensorList tensors,
    int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "stack expects a non-empty TensorList");
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
}

Tensor NestedTensor_stack(TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "stack expects a non-empty TensorList");
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat(get_stack_inputs(tensors, dim), dim);
}

Tensor& NestedTensor_cat_out(Tensor& result, TensorList tensors, int64_t dim) {
  auto tmp = at::cat(tensors, dim);
  result.copy_(tmp);
  return result;
}

Tensor NestedTensor_cat(TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "Cannot cat an empty list.");
  auto nested_dim_0 = get_nested_tensor_impl(tensors[0])->nested_dim();
  auto dim_0 = get_nested_tensor_impl(tensors[0])->dim();
  // TORCH_CHECK(dim == 0, "cat currently only supports dim set to 0.")
  for (size_t i = 1; i < tensors.size(); i++) {
    TORCH_CHECK(
        nested_dim_0 == get_nested_tensor_impl(tensors[i])->nested_dim(),
        "Nested dimension of NestedTensors must match for cat to succeed.");
    TORCH_CHECK(
        dim_0 == get_nested_tensor_impl(tensors[i])->dim(),
        "Dimension of NestedTensors must match for cat to succeed.");
  }
  if (dim == 0) {
    std::vector<TensorNode> result;
    for (size_t i = 0; i < tensors.size(); i++) {
      auto unbound = get_nested_tensor_structure(tensors[i]).unbind();
      for (size_t j = 0; j < unbound.size(); j++) {
        result.push_back(unbound[j]);
      }
    }
    return wrap_tensor_node(TensorNode(std::move(result)));
  }
  std::vector<std::vector<at::Tensor>> candidates;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto unbound = tensors[i].unbind();
    for (size_t j = 0; j < unbound.size(); j++) {
      if (j >= candidates.size()) {
        candidates.resize(j + 1);
      }
      candidates[j].push_back(unbound[j]);
    }
  }
  std::vector<TensorNode> result;
  for (size_t i = 0; i < candidates.size(); i++) {
    auto tmp = at::cat(TensorList(candidates[i]), dim - 1);
    if (is_nested_tensor_impl(tmp)) {
      result.push_back(get_nested_tensor_structure(tmp));
    } else {
      result.push_back(TensorNode(std::move(tmp)));
    }
  }
  return wrap_tensor_node(TensorNode(std::move(result)));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  m.impl_UNBOXED("conv2d", NestedTensor_conv2d);
  m.impl_UNBOXED("batch_norm", NestedTensor_batch_norm);
  m.impl_UNBOXED("max_pool2d", NestedTensor_max_pool2d);
  m.impl_UNBOXED("dropout", NestedTensor_dropout);
  m.impl_UNBOXED("dropout_", NestedTensor_dropout_);
  m.impl_UNBOXED("embedding", NestedTensor_embedding);
  m.impl_UNBOXED("add_.Tensor", NestedTensor_add_);
  m.impl_UNBOXED("any", NestedTensor_any);
  m.impl_UNBOXED("all", NestedTensor_all);
  m.impl_UNBOXED("_log_softmax", NestedTensor__log_softmax);
  m.impl_UNBOXED("reshape", NestedTensor_reshape);
  m.impl_UNBOXED("transpose.int", NestedTensor_transpose);
  m.impl_UNBOXED("softmax.int", NestedTensor_softmax);
  m.impl_UNBOXED("layer_norm", NestedTensor_layer_norm);
  m.impl_UNBOXED("matmul", NestedTensor_matmul);
  m.impl_UNBOXED("matmul.out", NestedTensor_matmul_out);
  m.impl_UNBOXED("pin_memory", NestedTensor_pin_memory);
  m.impl_UNBOXED("flatten.using_ints", NestedTensor_flatten);
  m.impl_UNBOXED("stack", NestedTensor_stack);
  m.impl_UNBOXED("stack.out", NestedTensor_stack_out);
  m.impl_UNBOXED("cat", NestedTensor_cat);
  m.impl_UNBOXED("cat.out", NestedTensor_cat_out);
}
} // namespace at
