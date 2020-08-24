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
  if (is_packed(self)) {
#ifdef TRACEPACKED
    std::cout << "calling packed reshape" << std::endl;
#endif
    auto self_structure = get_nested_tensor_structure(self);
    auto self_buffer = (*self_structure.buffer());
    return wrap_tensor_node(TensorNode(
        map(
            [target_shape](const at::Tensor t) {
              return at::reshape(t, IntArrayRef(target_shape));
            },
            self_structure),
        std::move(self_buffer)));
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
  if (is_packed(self)) {
#ifdef TRACEPACKED
    std::cout << "calling packed transpose" << std::endl;
#endif
    auto self_structure = get_nested_tensor_structure(self);
    auto self_buffer = (*self_structure.buffer());
    return wrap_tensor_node(TensorNode(
        map(
            [dim0, dim1, nested_dim](const at::Tensor t) {
              return at::transpose(t, dim0 - nested_dim, dim1 - nested_dim);
            },
            self_structure),
        std::move(self_buffer)));
  }
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
  AutoGradMode autogradmode(false);
  auto impl_self = get_nested_tensor_impl(self);
  auto structure_self = get_nested_tensor_structure(self);
  if (is_nested_tensor_impl(other)) {
    auto impl_other = get_nested_tensor_impl(other);
    auto structure_other = get_nested_tensor_structure(other);
    std::cout << self.dim() << std::endl;
    std::cout << other.dim() << std::endl;
    std::cout << *impl_self->opt_sizes()[0] << std::endl;
    std::cout << *impl_other->opt_sizes()[0] << std::endl;
    std::cout << *impl_self->opt_sizes()[1] << std::endl;
    std::cout << *impl_other->opt_sizes()[1] << std::endl;
    std::cout << *impl_self->opt_sizes()[3] << std::endl;
    std::cout << *impl_other->opt_sizes()[2] << std::endl;
    std::cout << (*impl_self->opt_sizes()[0] == *impl_other->opt_sizes()[0])
              << std::endl;
    std::cout << (*impl_self->opt_sizes()[1] == *impl_other->opt_sizes()[1])
              << std::endl;
    std::cout << (*impl_self->opt_sizes()[3] == *impl_other->opt_sizes()[2])
              << std::endl;
    if (structure_self.buffer() && structure_other.buffer() &&
        self.dim() == 4 && other.dim() == 4 && impl_self->opt_sizes()[0] &&
        impl_other->opt_sizes()[0] && impl_self->opt_sizes()[1] &&
        impl_other->opt_sizes()[1] && impl_self->opt_sizes()[3] &&
        impl_other->opt_sizes()[2] &&
        (*impl_self->opt_sizes()[0] == *impl_other->opt_sizes()[0]) &&
        (*impl_self->opt_sizes()[1] == *impl_other->opt_sizes()[1]) &&
        (*impl_self->opt_sizes()[3] == *impl_other->opt_sizes()[2])) {
#ifdef TRACEPACKED
      std::cout << "calling packed NT x NT matmul" << std::endl;
#endif
      SizeNode new_nested_size = map(
          [&](c10::List<int64_t> self_size, c10::List<int64_t> other_size) {
            c10::List<int64_t> new_size{
                self_size[0], self_size[1], other_size[2]};
            std::cout << "self_size[0]: " << self_size[0];
            std::cout << " - self_size[1]: " << self_size[1];
            std::cout << " - other_size[2]: " << other_size[2] << std::endl;
            return std::move(new_size);
          },
          impl_self->nested_size(),
          impl_other->nested_size());
      TensorNode transposed = 
          map(
              [](at::Tensor t) {
                auto ti = t.transpose(1, 2).transpose(0, 1);
                std::cout << "ti.sizes(): " << ti.sizes() << std::endl;
                std::cout << "ti.strides(): " << ti.strides() << std::endl;
                return ti;
              },
              structure_other);
      TensorNode transposed0 = 
          map(
              [](at::Tensor t) {
                auto ai = t;
                std::cout << "ai.sizes(): " << ai.sizes() << std::endl;
                std::cout << "ai.strides(): " << ai.strides() << std::endl;
                return ai;
              },
              structure_self);
      std::cout << "structure_self_buffer.sizes(): " << (*structure_self.buffer()).sizes() << std::endl;
      std::cout << "structure_other_buffer.sizes(): " << (*structure_other.buffer()).sizes() << std::endl;
      auto matmul_self_buffer =
          (*structure_self.buffer()).reshape({
             // ((*impl_self->opt_sizes()[0]) * (*impl_self->opt_sizes()[1])), 
             -1,
             // (*impl_self->opt_sizes()[0]), 
              1, *impl_self->opt_sizes()[3]});
      auto matmul_other_buffer =
          (*structure_other.buffer())
              .reshape({
                 // ((*impl_self->opt_sizes()[0]) * (*impl_self->opt_sizes()[1])), 
                 -1, 
                 // (*impl_self->opt_sizes()[0]), 
                  1, *impl_self->opt_sizes()[3]}).transpose(1, 2);
      std::cout << "matmul_self_buffer.sizes(): " << matmul_self_buffer.sizes() << std::endl;
      std::cout << "matmul_other_buffer.sizes(): " << matmul_other_buffer.sizes() << std::endl;
      auto new_buffer =
          at::matmul(matmul_self_buffer, matmul_other_buffer);
      std::cout << "new_buffer.sizes(): " << new_buffer.sizes() << std::endl;
      new_buffer = new_buffer.reshape({-1});
      return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
          std::move(new_buffer), new_nested_size));
    }
    return map_nested_tensor(
        [](Tensor tensor, Tensor other) { return at::matmul(tensor, other); },
        self,
        other);
  }
  if (structure_self.buffer()) {
    if (self.dim() == 3 && other.dim() == 2 && impl_self->opt_sizes()[0] &&
        impl_self->opt_sizes()[2] &&
        impl_self->opt_sizes()[self.dim() - 1] == other.size(self.dim() - 2)) {
#ifdef TRACEPACKED
      std::cout << "calling packed NT x T matmul" << std::endl;
#endif
      SizeNode new_nested_size = map(
          [&](c10::List<int64_t> self_size) {
            c10::List<int64_t> new_size{self_size[0], other.size(1)};
            return std::move(new_size);
          },
          impl_self->nested_size());
      return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
          at::matmul(
              (*structure_self.buffer()).reshape({-1, other.size(0)}), other)
              .reshape(-1),
          new_nested_size));
    }
  }
  return map_nested_tensor(
      [&other](Tensor tensor) { return at::matmul(tensor, other); }, self);
}

Tensor& NestedTensor_matmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  AutoGradMode autogradmode(false);
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
  nt_impl(m, "embedding", NestedTensor_embedding);
  nt_impl(m, "any", NestedTensor_any);
  nt_impl(m, "all", NestedTensor_all);
  nt_impl(m, "_log_softmax", NestedTensor__log_softmax);
  nt_impl(m, "reshape", NestedTensor_reshape);
  nt_impl(m, "transpose.int", NestedTensor_transpose);
  nt_impl(m, "softmax.int", NestedTensor_softmax);
  nt_impl(m, "layer_norm", NestedTensor_layer_norm);
  // nt_impl(m, "matmul", no_bw(TORCH_FN(NestedTensor_matmul);
  nt_impl(m, "matmul", NestedTensor_matmul);
  nt_impl(m, "matmul.out", NestedTensor_matmul_out);
  nt_impl(m, "pin_memory", NestedTensor_pin_memory);
  nt_impl(m, "flatten.using_ints", NestedTensor_flatten);
  nt_impl(m, "stack", NestedTensor_stack);
  nt_impl(m, "stack.out", NestedTensor_stack_out);
  nt_impl(m, "cat", NestedTensor_cat);
  nt_impl(m, "cat.out", NestedTensor_cat_out);
}
} // namespace at
