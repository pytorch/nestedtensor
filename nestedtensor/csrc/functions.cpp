#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  return wrap_tensor_node(
      map([&](const at::Tensor t) { return at::dropout(t, p, train); },
          get_nested_tensor_structure(input)));
}

Tensor& NestedTensor_dropout_(Tensor& input, double p, bool train) {
  apply(
      [&](at::Tensor t) { return at::dropout_(t, p, train); },
      get_nested_tensor_structure(input));
  return input;
}

Tensor NestedTensor_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  return wrap_tensor_node(map(
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
      get_nested_tensor_structure(input)));
}

Tensor NestedTensor_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  auto self_impl = get_nested_tensor_impl(self);
  auto tensor_node = get_nested_tensor_structure(self);

  if (is_tensor_shape(self)) {
    std::vector<at::Tensor> tensors;
    for (auto tn : tensor_node.unbind()) {
      tensors.push_back(tn.payload());
    }

    auto res_ = at::max_pool2d(
        at::stack(tensors), kernel_size, stride, padding, dilation, ceil_mode);
    std::vector<at::Tensor> res = res_.unbind();
    std::vector<TensorNode> result;
    for (size_t i = 0; i < res.size(); i++) {
      result.push_back(TensorNode(std::move(res[i])));
    }
    return NestedTensorImpl(TensorNode(std::move(result)))
        .to_nested_tensor(self_impl->nested_dim() - 1);
  }

  return wrap_tensor_node(map(
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
      get_nested_tensor_structure(self)));
}

Tensor NestedTensor_batch_norm(
    const Tensor& input,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    const Tensor& running_mean /* optional */,
    const Tensor& running_var /* optional */,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  return wrap_tensor_node(map(
      [&](at::Tensor t) {
        auto result = at::batch_norm(
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
        return result;
      },
      get_nested_tensor_structure(input)));
}

Tensor NestedTensor_sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::sum(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({0}, *dtype);
    }
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors.vec());
  return at::sum(all_tensor, dtype);
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
  return wrap_tensor_node(map(
      [target_shape](const at::Tensor t) {
        return at::reshape(t, IntArrayRef(target_shape));
      },
      get_nested_tensor_structure(self)));
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
  return wrap_tensor_node(map(
      [dim0, dim1, nested_dim](const at::Tensor t) {
        return at::transpose(t, dim0 - nested_dim, dim1 - nested_dim);
      },
      get_nested_tensor_structure(self)));
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
  return wrap_tensor_node(map(
      [dim, nested_dim, dtype](const at::Tensor t) {
        return at::softmax(t, dim - nested_dim, dtype);
      },
      get_nested_tensor_structure(input)));
}

Tensor NestedTensor_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
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
  return wrap_tensor_node(map(
      [normalized_shape, &weight, &bias, eps](const at::Tensor t) {
        return at::layer_norm(t, normalized_shape, weight, bias, eps, true);
      },
      input_data->get_structure()));
}

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  if (is_nested_tensor_impl(other)) {
    apply(
        [alpha](Tensor& self, Tensor& other) { self.add_(other, alpha); },
        get_nested_tensor_structure(self),
        get_nested_tensor_structure(other));
    return self;
  }
  apply(
      [&other, alpha](at::Tensor& self) { return self.add_(other, alpha); },
      get_nested_tensor_structure(self));
  return self;
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
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  auto self_impl = get_nested_tensor_impl(input_);
  return wrap_tensor_node(
      map([&](Tensor a) { return at::_log_softmax(a, dim_, half_to_float); },
          self_impl->get_structure()));
}

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(other)) {
    return wrap_tensor_node(map(
        [](Tensor tensor, Tensor other) { return at::matmul(tensor, other); },
        get_nested_tensor_structure(self),
        get_nested_tensor_structure(other)));
  }
  return wrap_tensor_node(
      map([&other](Tensor tensor) { return at::matmul(tensor, other); },
          get_nested_tensor_structure(self)));
}

Tensor& NestedTensor_matmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  apply(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        return at::matmul_out(result, tensor, other);
      },
      get_nested_tensor_structure(result),
      get_nested_tensor_structure(self),
      get_nested_tensor_structure(other));
  return result;
}

Tensor NestedTensor_pin_memory(const Tensor& self) {
  return wrap_tensor_node(
      map([](Tensor tensor) { return at::native::pin_memory(tensor); },
          get_nested_tensor_structure(self)));
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
  return wrap_tensor_node(map(
      [start_dim, end_dim, nested_dim](at::Tensor tensor) {
        return at::flatten(
            tensor, start_dim - nested_dim, end_dim - nested_dim);
      },
      self_data->get_structure()));
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
  m.impl_UNBOXED("sum", NestedTensor_sum);
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
