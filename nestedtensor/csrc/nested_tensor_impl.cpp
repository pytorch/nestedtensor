#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;
using namespace c10;

TensorNode _unbind_tensors(TensorNode structure) {
  std::vector<TensorNode> result_nodes;
  if (structure.is_leaf()) {
    for (at::Tensor tensor : structure.payload().unbind()) {
      result_nodes.emplace_back(TensorNode(std::move(tensor)));
    }
  } else {
    for (TensorNode child : structure.unbind()) {
      result_nodes.emplace_back(_unbind_tensors(child));
    }
  }
  return TensorNode(std::move(result_nodes));
}

NestedTensorImpl::NestedTensorImpl(std::shared_ptr<NestedTensorStorage> storage)
    : TensorImpl(
          c10::DispatchKeySet({NestedTensorKey}),
          storage->dtype(),
          storage->device()),
      _storage(storage) {
  remove_autograd_key();
  key_set_ = key_set_ - c10::DispatchKeySet({DispatchKey::ADInplaceOrView});
}

inline TensorNode _squeeze_nested_dim(TensorNode structure, int64_t dim) {
  return squeeze(structure, dim, false);
}

int64_t NestedTensor_size_int(const Tensor& self, int64_t dim) {
  std::vector<c10::optional<int64_t>> size =
      get_nested_tensor_impl(self)->opt_sizes();
  if (size[dim]) {
    return *(size[dim]);
  }
  throw std::runtime_error(
      "NestedTensor size at dim is not Tensor shape compliant.");
}

IntArrayRef NestedTensorImpl::strides() const {
  TORCH_CHECK(
      false,
      "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
  std::vector<int64_t> strides;
  return IntArrayRef(strides);
}

int64_t nt_size(Tensor tensor, int64_t dim) {
  auto impl = get_nested_tensor_impl(tensor);
  std::vector<c10::optional<int64_t>> size = impl->opt_sizes();
  if (size[dim]) {
    return *(size[dim]);
  }
  throw std::runtime_error(
      "NestedTensor size at dim is not Tensor shape compliant.");
}

at::Tensor wrap_tensor_node(TensorNode&& result) {
  if (result.is_leaf()) {
    return result.payload();
  }
  ListStorage* ls = new ListStorage(std::move(result));
  NestedTensorStorage* ls_base = dynamic_cast<NestedTensorStorage*>(ls);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::shared_ptr<NestedTensorStorage>(ls_base));
}

std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode> input) {
  std::vector<at::Tensor> result;
  for (size_t i = 0; i < input.size(); i++) {
    result.push_back(wrap_tensor_node(std::move(input[i])));
  }
  return result;
}

at::Tensor wrap_buffer(at::Tensor&& buffer, SizeNode nested_size) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  if (nested_size.is_leaf()) {
    return buffer.reshape(IntArrayRef(nested_size.payload()));
  }
  PackedStorage* ps = new PackedStorage(std::move(buffer), nested_size);
  NestedTensorStorage* ps_base = dynamic_cast<NestedTensorStorage*>(ps);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::shared_ptr<NestedTensorStorage>(ps_base));
}

at::Tensor wrap_buffer(
    at::Tensor&& buffer,
    EfficientSizeNode efficient_nested_size,
    EfficientSizeNode efficient_nested_stride) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  TORCH_CHECK(
      efficient_nested_size.height() > 0,
      "Internal error: expected nested_size of non-zero height.");
  TORCH_CHECK(
      efficient_nested_stride.height() > 0,
      "Internal error: expected nested_size of non-zero height.");
  PackedStorage* ps = new PackedStorage(
      std::move(buffer), efficient_nested_size, efficient_nested_stride);
  NestedTensorStorage* ps_base = dynamic_cast<NestedTensorStorage*>(ps);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::shared_ptr<NestedTensorStorage>(ps_base));
}

Tensor NestedTensor_contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  PackedStorage* ps = new PackedStorage(get_nested_tensor_structure(self));
  NestedTensorStorage* ps_base = dynamic_cast<NestedTensorStorage*>(ps);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::shared_ptr<NestedTensorStorage>(ps_base));
}

bool NestedTensor_is_pinned(const Tensor& self) {
  return get_nested_tensor_impl(self)->is_pinned();
}

std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  auto _data = get_nested_tensor_impl(self);
  dim = at::maybe_wrap_dim(dim, get_dim(self));
  auto node = _data->get_structure();
  if (dim == 0) {
    return wrap_tensor_node(node.unbind());
  }
  std::vector<std::vector<TensorNode>> unbound;
  for (auto child : node.unbind()) {
    std::vector<at::Tensor> tmp =
        at::unbind(wrap_tensor_node(std::move(child)), dim - 1);
    for (size_t j = 0; j < tmp.size(); j++) {
      if (j >= unbound.size()) {
        unbound.resize(j + 1);
      }
      unbound[j].push_back(TensorNode(std::move(tmp[j])));
    }
  }
  std::vector<TensorNode> result;
  for (size_t i = 0; i < unbound.size(); i++) {
    result.push_back(TensorNode(std::move(unbound[i])));
  }
  return wrap_tensor_node(result);
}

Tensor NestedTensor_select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = get_dim(self);
  dim = maybe_wrap_dim(dim, ndim);
  if (dim != 0) {
    TORCH_CHECK_INDEX(false, "select() only supports dim == 0 for now.");
  }
  auto tmp = get_nested_tensor_structure(self).unbind()[index];
  return wrap_tensor_node(std::move(tmp));
}

Tensor NestedTensor_to_nested_tensor(
    at::Tensor input,
    c10::optional<int64_t> dim_) {
  int64_t dim = 0;
  if (dim_) {
    dim = *dim_;
    dim = maybe_wrap_dim(*dim_, get_dim(input) + 1);
  }
  TORCH_CHECK(
      dim <= get_dim(input),
      "target nested dimension needs to be equal or less than to input dimension");
  // if dim < nested_dim() the NestedTensor is already nested
  // up to the given dimension.
  if (is_nested_tensor_impl(input) && dim >= get_nested_dim(input)) {
    TensorNode unbound = _unbind_tensors(get_nested_tensor_structure(input));
    for (int64_t i = 0; i < (dim - get_nested_dim(input)); i++) {
      unbound = _unbind_tensors(unbound);
    }
    return wrap_tensor_node(std::move(unbound));
  }
  if (!is_nested_tensor_impl(input) && dim > 0) {
    std::vector<TensorNode> unbound_nodes;
    for (at::Tensor t : input.unbind()) {
      unbound_nodes.push_back(TensorNode(std::move(t)));
    }
    TensorNode unbound(std::move(unbound_nodes));
    for (int64_t i = 1; i < dim; i++) {
      unbound = _unbind_tensors(unbound);
    }
    return wrap_tensor_node(std::move(unbound));
  }
  return input;
}

// TODO: There are unanswered questions
// around 0-numel NestedTensors as maybe brought about by
// t[:, out_of_bounds:, :]
Tensor NestedTensor_slice(
    const Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start_,
    c10::optional<int64_t> end_,
    int64_t step) {
  int64_t start;
  if (start_) {
    start = *start_;
  } else {
    start = 0;
  }
  int64_t end;
  if (end_) {
    end = *end_;
  } else {
    end = 9223372036854775807;
  }
  int64_t ndim = get_dim(self);
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  if (dim != 0) {
    TORCH_CHECK_INDEX(false, "slice() only supports dim == 0 for now.");
  }
  // TODO: support negative strides
  TORCH_CHECK(step >= 1, "slice step must be positive for now.");
  int64_t sizes_0 = nt_size(self, 0);
  if (start < 0) {
    start += sizes_0;
  }
  if (end < 0) {
    end += sizes_0;
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes_0) {
    start = sizes_0;
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes_0) {
    end = sizes_0;
  }
  std::vector<at::Tensor> unbound = at::unbind(self, 0);
  std::vector<TensorNode> new_tensor_nodes;
  for (int64_t i = start; i < end; i += step) {
    if (is_nested_tensor_impl(unbound[i])) {
      new_tensor_nodes.push_back(get_nested_tensor_structure(unbound[i]));
    } else {
      new_tensor_nodes.push_back(TensorNode(std::move(unbound[i])));
    }
  }
  auto result = wrap_tensor_node(TensorNode(std::move(new_tensor_nodes)));
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& NestedTensor_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  apply_nested_tensor(
      [](at::Tensor& self, at::Tensor& source) { return self.copy_(source); },
      self,
      src);
  return self;
}

Tensor _NestedTensor_squeeze_(Tensor self, c10::optional<int64_t> dim_) {
  auto self_impl = get_nested_tensor_impl(self);
  if (!dim_) {
    // TODO: First dimension is always ignored.
    // We could decide to return a Tensor if the 0th
    // dimension can be squeezed.
    auto init_sizes = self_impl->opt_sizes();
    for (size_t i = 0; i < init_sizes.size() - 1; i++) {
      int64_t index = init_sizes.size() - i - 1;
      c10::optional<int64_t> s = init_sizes[index];
      if (s && ((*s) == 1)) {
        self = _NestedTensor_squeeze_(self, index);
      }
    }
    return self;
  }
  int64_t dim = at::maybe_wrap_dim(*dim_, get_dim(self));
  TORCH_CHECK(dim > 0, "Cannot squeeze first dimension.");
  TORCH_CHECK(
      ((get_nested_tensor_impl(self)->opt_sizes()[dim]) &&
       ((*(get_nested_tensor_impl(self)->opt_sizes()[dim])) == 1)),
      "Given dimension is either undefined or not a singleton.");
  if (dim < get_nested_tensor_impl(self)->nested_dim()) {
    return wrap_tensor_node(
        _squeeze_nested_dim(self_impl->get_structure(), dim));
  }
  int64_t height = self_impl->get_structure().height();
  return map_nested_tensor(
      [dim, height](at::Tensor tensor) { return tensor.squeeze(dim - height); },
      self);
}

Tensor& NestedTensor_squeeze_(Tensor& self) {
  self = _NestedTensor_squeeze_(self, c10::nullopt);
  return self;
}

Tensor& NestedTensor_squeeze__dim(Tensor& self, int64_t dim) {
  self = _NestedTensor_squeeze_(self, dim);
  return self;
}

Tensor NestedTensor_squeeze_dim(const Tensor& self, int64_t dim) {
  dim = at::maybe_wrap_dim(dim, get_dim(self));
  auto self_impl = get_nested_tensor_impl(self);
  int64_t nested_dim = self_impl->nested_dim();
  TORCH_CHECK(dim > 0, "Cannot squeeze first dimension.");
  TORCH_CHECK(dim >= nested_dim, "Cannot squeeze nested dimension.");
  TORCH_CHECK(
      ((self_impl->opt_sizes()[dim]) &&
       ((*(self_impl->opt_sizes()[dim])) == 1)),
      "Given dimension is either undefined or not a singleton.");
  return map_nested_tensor(
      [dim, nested_dim](at::Tensor tensor) {
        return tensor.squeeze(dim - nested_dim);
      },
      self);
}

Tensor NestedTensor_squeeze(const Tensor& self) {
  TORCH_CHECK(false, "squeeze(Tensor) is currently not implemented.");
}

Tensor NestedTensor_unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, get_dim(self) + 1);
  if (dim == 0) {
    std::vector<TensorNode> one_node;
    one_node.push_back(get_nested_tensor_structure(self));
    return wrap_tensor_node(TensorNode(std::move(one_node)));
  }
  std::vector<TensorNode> result_nodes;
  auto unbound = self.unbind(0);
  for (size_t i = 0; i < unbound.size(); i++) {
    result_nodes.push_back(
        get_nested_tensor_structure(at::unsqueeze(unbound[i], dim - 1)));
  }
  return wrap_tensor_node(TensorNode(std::move(result_nodes)));
}

Tensor NestedTensor_serialize_nested_size(const Tensor& tensor) {
  auto nt_impl = get_nested_tensor_impl(tensor);
  std::vector<int64_t> out;
  return torch::tensor(torch::nested_tensor::serialize(nt_impl->nested_size()));
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "contiguous", NestedTensor_contiguous);
  nt_impl(m, "copy_", NestedTensor_copy_);
  nt_impl(m, "is_pinned", NestedTensor_is_pinned);
  nt_impl(m, "select.int", NestedTensor_select);
  nt_impl(m, "serialize_nested_size", NestedTensor_serialize_nested_size);
  nt_impl(m, "size.int", NestedTensor_size_int);
  nt_impl(m, "slice.Tensor", NestedTensor_slice);
  nt_impl(m, "squeeze", NestedTensor_squeeze);
  nt_impl(m, "squeeze.dim", NestedTensor_squeeze_dim);
  nt_impl(m, "squeeze_", NestedTensor_squeeze_);
  nt_impl(m, "squeeze_.dim", NestedTensor_squeeze__dim);
  nt_impl(m, "unbind.int", NestedTensor_unbind);
  nt_impl(m, "unsqueeze", NestedTensor_unsqueeze);
}
} // namespace at
