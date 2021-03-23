#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

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
  if (weight && bias) {
    return map_nested_tensor(
        [normalized_shape, eps](const at::Tensor t, Tensor w, Tensor b) {
          return at::layer_norm(t, normalized_shape, w, b, eps, true);
        },
        input,
        *weight,
        *bias);
  }
  TORCH_CHECK(!weight && !bias, "Either both weight and bias are used or not.");
  return map_nested_tensor(
      [normalized_shape, eps](const at::Tensor t) {
        return at::layer_norm(
            t, normalized_shape, c10::nullopt, c10::nullopt, eps, true);
      },
      input);
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
  // XXX: Write test that checks for flatten autograd support.
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
    TensorList tensors,
    int64_t dim,
    Tensor& result) {
  TORCH_CHECK(tensors.size() > 0, "stack expects a non-empty TensorList");
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
}

Tensor NestedTensor_stack(TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "stack expects a non-empty TensorList");
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat(get_stack_inputs(tensors, dim), dim);
}

Tensor& NestedTensor_cat_out(TensorList tensors, int64_t dim, Tensor& result) {
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

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "embedding", NestedTensor_embedding);
  nt_impl(m, "any", NestedTensor_any);
  nt_impl(m, "all", NestedTensor_all);
  nt_impl(m, "_log_softmax", NestedTensor__log_softmax);
  nt_impl(m, "layer_norm", NestedTensor_layer_norm);
  nt_impl(m, "pin_memory", NestedTensor_pin_memory);
  nt_impl(m, "flatten.using_ints", NestedTensor_flatten);
  nt_impl(m, "stack", NestedTensor_stack);
  nt_impl(m, "stack.out", NestedTensor_stack_out);
  nt_impl(m, "cat", NestedTensor_cat);
  nt_impl(m, "cat.out", NestedTensor_cat_out);
}

} // namespace at
