#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

IntArrayRef NestedTensorImpl::sizes() const {
  return IntArrayRef(_sizes);
}

int64_t NestedTensorImpl::size(int64_t dim) const {
  std::vector<c10::optional<int64_t>> size = _data.sizes();
  if (size[dim]) {
    return *(size[dim]);
  }
  throw std::runtime_error("NestedTensor size at dim is not Tensor shape compliant.");
}

IntArrayRef NestedTensorImpl::strides() const {
  throw std::runtime_error("NestedTensor stride is not implemented.");
}

Tensor NestedTensor_contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  auto nt = get_nested_tensor(self).contiguous();
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_) {
  auto impl_data = get_nested_tensor_impl(tensor)->_data;
  if (!dim_) {
    return impl_data.to_tensor();
  }
  int64_t dim = maybe_wrap_dim((*dim_), impl_data.dim());
  if (dim == 0) {
    return impl_data.to_tensor();
  }
  // If dim is bigger than nested_dim the NestedTensor is already
  // of Tensor for dimensions bigger than the given.
  if (impl_data.nested_dim() == 1) {
    return tensor;
  }
  // At this point nested_dim is at least 2. That means any unbind
  // operation of a child must yield NestedTensors.
  // If dim is 1 then we'll apply to_tensor(0) to the children and must expect
  // Tensors.
  std::vector<at::Tensor> unbound = at::unbind(tensor, 0);
  std::vector<TensorNode> result;
  for (Tensor child : unbound) {
    auto ci = NestedTensor_to_tensor(child, dim - 1);
    if (is_nested_tensor_impl(ci)) {
      auto s = get_nested_tensor(ci).get_structure();
      result.push_back(TensorNode(std::move(s)));
    } else {
      // TODO: If it's a NestedTensor instance get the structure
      result.push_back(TensorNode(std::move(ci)));
    }
  }
  return at::detail::make_tensor<at::NestedTensorImpl>(
      NestedTensor(TensorNode(std::move(result))));
}

bool NestedTensor_is_pinned(const Tensor& self) {
  return get_nested_tensor(self).is_pinned();
}

std::vector<at::Tensor> NestedTensor_unbind(const at::Tensor &self, int64_t dim) {
  auto _data = get_nested_tensor(self);
  dim = at::maybe_wrap_dim(dim, _data.dim());
  auto node = _data.get_structure();
  auto nested_dim = _data.nested_dim();
  if (nested_dim == 1) {
    if (dim == 0) {
      std::vector<at::Tensor> result;
      for (const auto& child : node.unbind()) {
        result.push_back(child.payload());
      }
      return result;
    } else {
      int64_t dim_max_size = 0;
      for (const auto& child : node.unbind()) {
        int64_t dim_size = child.payload().size(dim - 1);
        dim_max_size = dim_max_size > dim_size ? dim_max_size : dim_size;
      }
      std::vector<std::vector<TensorNode>> unbound;
      unbound.resize(dim_max_size);
      for (const auto& child : node.unbind()) {
        std::vector<at::Tensor> unbound_tensors =
            at::unbind(child.payload(), dim - 1);
        for (size_t i = 0; i < unbound_tensors.size(); i++) {
          unbound[i].push_back(TensorNode(std::move(unbound_tensors[i])));
        }
      }
      std::vector<at::Tensor> result;
      for (size_t i = 0; i < unbound.size(); i++) {
        TensorNode tmp = TensorNode(std::move(unbound[i]));
        result.push_back(at::detail::make_tensor<NestedTensorImpl>(NestedTensor(std::move(tmp))));
      }
      return result;
    }
  }
  std::vector<at::Tensor> unbound_thp;
  for (auto child : node.unbind()) {
    unbound_thp.push_back(at::detail::make_tensor<NestedTensorImpl>(NestedTensor(std::move(child))));
  }
  if (dim == 0) {
    return unbound_thp;
  }
  std::vector<std::vector<TensorNode>> unbound;
  for (size_t i = 0; i < unbound_thp.size(); i++) {
    std::vector<at::Tensor> tmp = unbound_thp[i].unbind(dim - 1);
    for (size_t j = 0; j < tmp.size(); j++) {
      if (unbound.size() >= j) {
        unbound.resize(j + 1);
      }
      unbound[j].push_back(TensorNode(std::move(tmp[j])));
    }
  }
  std::vector<at::Tensor> result;
  for (size_t i = 0; i < unbound.size(); i++) {
    result.push_back(at::detail::make_tensor<NestedTensorImpl>(
        NestedTensor(TensorNode(std::move(unbound[i])))));
  }
  return result;
}

Tensor NestedTensor_select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  if (dim == 0) {
    TORCH_CHECK_INDEX(false, "select() only supports dim == 0 for now.");
  }
  TensorNode tn = get_nested_tensor(self).get_structure().unbind()[index];
  torch::nested_tensor::NestedTensor nt = torch::nested_tensor::NestedTensor(
      std::move(tn));
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  auto self_impl = get_nested_tensor_impl(self);
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
  auto self_impl = get_nested_tensor_impl(self)->_data;
  if (self_impl.numel() == 0) {
    // XXX: self.options doesn't work here because
    // we don't want a Tensor backed by a NestedTensor
    Tensor result = at::empty({0}, at::kBool); //, self.options());
    result.fill_(1);
    return result;
  }
  auto map_all = flatten(map(
        [](at::Tensor tensor) { return tensor.all(); },
          self_impl.get_structure()));
  at::Tensor gathered = at::empty(
      {static_cast<int64_t>(map_all.size())}, at::kBool); //, self.options());
  for (size_t i = 0; i < map_all.size(); i++) {
    gathered[i] = map_all[i];
  }
  return gathered.all();
}

Tensor NestedTensor_any(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self)->_data;
  if (self_impl.numel() == 0) {
    // XXX: self.options doesn't work here because
    // we don't want a Tensor backed by a NestedTensor
    Tensor result = at::empty({0}, at::kBool); //, self.options());
    result.fill_(1);
    return result;
  }
  auto map_any = flatten(map(
        [](at::Tensor tensor) { return tensor.any(); },
          self_impl.get_structure()));
  at::Tensor gathered = at::empty(
      {static_cast<int64_t>(map_any.size())}, at::kBool); //, self.options());
  for (size_t i = 0; i < map_any.size(); i++) {
    gathered[i] = map_any[i];
  }
  return gathered.any();
}

Tensor NestedTensor_clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto self_impl = get_nested_tensor_impl(src);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([&optional_memory_format](Tensor a) {
          return at::clone(a, optional_memory_format);
          }, 
          self_impl->_data.get_structure()));
}

Tensor NestedTensor__log_softmax(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  auto self_impl = get_nested_tensor_impl(input_);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([&](Tensor a) {
          return at::_log_softmax(a, dim_, half_to_float);
          }, 
          self_impl->_data.get_structure()));
}

Tensor& NestedTensor_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  auto self_impl = get_nested_tensor_impl(self);
  auto src_impl = get_nested_tensor_impl(src);
  self_impl->_data.copy_(src_impl->_data);
  return self;
}

Tensor& NestedTensor_squeeze_(Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  self_impl->_data.squeeze_(c10::nullopt);
  return self;
}

Tensor& NestedTensor_squeeze__dim(Tensor& self, int64_t dim) {
  auto self_impl = get_nested_tensor_impl(self);
  self_impl->_data.squeeze_(dim);
  return self;
}

Tensor NestedTensor_squeeze(const Tensor& self) {
  auto new_tensor = NestedTensor_clone(self, c10::nullopt);
  return NestedTensor_squeeze_(new_tensor);
}

Tensor NestedTensor_squeeze_dim(const Tensor& self, int64_t dim) {
  auto new_tensor = NestedTensor_clone(self, c10::nullopt);
  return NestedTensor_squeeze__dim(new_tensor, dim);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  m.impl_UNBOXED("clone", NestedTensor_clone);
  m.impl_UNBOXED("_log_softmax", NestedTensor__log_softmax);
  m.impl_UNBOXED("copy_", NestedTensor_copy_);
  m.impl_UNBOXED("squeeze_", NestedTensor_squeeze_);
  m.impl_UNBOXED("squeeze_.dim", NestedTensor_squeeze__dim);
  m.impl_UNBOXED("squeeze", NestedTensor_squeeze);
  m.impl_UNBOXED("squeeze.dim", NestedTensor_squeeze_dim);
  m.impl_UNBOXED("any", NestedTensor_any);
  m.impl_UNBOXED("all", NestedTensor_all);
  m.impl_UNBOXED("add_.Tensor", NestedTensor_add_);
  m.impl_UNBOXED("contiguous", NestedTensor_contiguous);
  m.impl_UNBOXED("is_pinned", NestedTensor_is_pinned);
  m.impl_UNBOXED("unbind.int", NestedTensor_unbind);
  m.impl_UNBOXED("select.int", NestedTensor_select);
}

}
