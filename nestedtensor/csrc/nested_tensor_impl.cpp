#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>

namespace torch {
namespace nested_tensor {

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  if (size.size() == 0) {
    return 1;
  }
  return size[0] * stride[0];
}

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
}

std::vector<c10::optional<int64_t>> NestedTensor::sizes() const {
  return construct_size(_nested_size);
}

c10::List<int64_t> _cont_stride(c10::List<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return c10::List<int64_t>(stride);
}

TensorNode build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size,
    const SizeNode& nested_stride) {
  c10::List<int64_t> split_sizes = flatten(
      map([](c10::List<int64_t> a,
             c10::List<int64_t> b) { return num_memory(a, b); },
          nested_size,
          nested_stride));
  std::vector<int64_t> nonzero_split_sizes;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      nonzero_split_sizes.push_back(split_sizes[i]);
    }
  }
  std::vector<at::Tensor> buffers_;
  if (nonzero_split_sizes.size() > 0) {
    buffers_ =
        at::split_with_sizes(buffer, c10::IntArrayRef(nonzero_split_sizes), 0);
  }
  std::vector<at::Tensor> buffers;
  int64_t index = 0;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      buffers.push_back(buffers_[index]);
      index++;
    } else {
      buffers.push_back(at::empty({}, buffer.options()));
    }
  }
  TensorNode tmp = unflatten(nested_size, c10::List<at::Tensor>(buffers));
  TensorNode result = map(
      [](at::Tensor buffer,
         c10::List<int64_t> size,
         c10::List<int64_t> stride) {
        return at::as_strided(
            buffer,
            c10::IntArrayRef(size.vec()),
            c10::IntArrayRef(stride.vec()));
      },
      tmp,
      nested_size,
      nested_stride);
  return result;
}

TensorNode build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size) {
  SizeNode nested_stride = map(
      [](c10::List<int64_t> size) { return _cont_stride(size); }, nested_size);
  return build_structure(buffer, nested_size, nested_stride);
}

SizeNode infer_nested_size(const TensorNode& _structure) {
  return map(
      [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
      _structure);
}

NestedTensor NestedTensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  TensorNode flat_structure =
      map([](at::Tensor tensor) { return tensor.reshape({-1}); }, _structure);
  auto tensors = flatten(flat_structure).vec();
  if (tensors.size() == 0) {
    return NestedTensor(at::ones({0}), _nested_size);
  }
  return NestedTensor(at::cat(tensors, 0), _nested_size);
}

at::Tensor _to_tensor(TensorNode node) {
  // TODO: Recursive stacking is expensive.
  if (node.is_leaf()) {
    return node.payload();
  }
  if (node.degree() == 0) {
    return at::empty({0});
  }
  std::vector<at::Tensor> flat;
  for (auto child : node.unbind()) {
    flat.push_back(_to_tensor(child));
  }
  return stack(flat);
}

at::Tensor NestedTensor::to_tensor() {
  // TODO: Not necessarily a view because of stack and reshape.
  std::vector<int64_t> new_size;
  for (const auto& si : sizes()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t at
      // this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  if (is_contiguous()) {
    return (*_buffer).reshape(at::IntArrayRef(new_size));
  }
  return _to_tensor(_structure);
}

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

NestedTensor NestedTensor::to_nested_tensor(c10::optional<int64_t> dim__) {
  int64_t dim_ = 0;
  if (dim__) {
    dim_ = *dim__;
  }
  int64_t dim = at::maybe_wrap_dim(dim_, this->dim());
  // if dim < nested_dim() the NestedTensor is already nested
  // up to the given dimension.
  if (dim >= nested_dim()) {
    TensorNode unbound = _unbind_tensors(_structure);
    for (int64_t i = 0; i < (dim - nested_dim()); i++) {
      unbound = _unbind_tensors(unbound);
    }
    return NestedTensor(std::move(unbound));
  }
  return *this;
}

NestedTensor::NestedTensor(TensorNode&& structure)
    : _structure(structure),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(infer_nested_size(_structure)) {}

// NOTE: It is assumed that structure is a tree of views
// of buffer.
// TODO: Add an explicit test for debug purposes.
NestedTensor::NestedTensor(at::Tensor&& buffer, TensorNode&& structure)
    : _buffer(buffer),
      _structure(structure),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(infer_nested_size(_structure)) {}

NestedTensor::NestedTensor(at::Tensor&& buffer, SizeNode nested_size)
    : _buffer(buffer),
      _structure(build_structure(*_buffer, nested_size)),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(nested_size) {}

// torch.Tensor methods
NestedTensor NestedTensor::copy_(
    const NestedTensor& source,
    bool non_blocking) {
  TORCH_CHECK(
      shape_matches(nested_size(), source.nested_size()),
      "self and source don't match in shape");
  if (_buffer && source.get_buffer()) {
    _buffer->copy_(*source.get_buffer());
    return *this;
  }
  if (_buffer) {
    NestedTensor cont_source = source.contiguous();
    _buffer->copy_(*cont_source.get_buffer());
    return *this;
  }
  auto result =
      map([](at::Tensor self, at::Tensor source) { return self.copy_(source); },
          _structure,
          source.get_structure());
  return *this;
}

inline TensorNode _squeeze_nested_dim(TensorNode structure, int64_t dim) {
  if (dim == 0) {
    return structure.children(0);
  }
  return TensorNode(_squeeze_nested_dim(structure, dim - 1));
}

NestedTensor NestedTensor::squeeze_(c10::optional<int64_t> dim_) {
  if (!dim_) {
    // TODO: First dimension is always ignored.
    // We could decide to return a Tensor if the 0th
    // dimension can be squeezed.
    auto init_sizes = sizes();
    for (size_t i = 0; i < init_sizes.size() - 1; i++) {
      int64_t index = init_sizes.size() - i - 1;
      c10::optional<int64_t> s = init_sizes[index];
      if (s && ((*s) == 1)) {
        this->squeeze_(index);
      }
    }
    return *this;
  }
  int64_t dim = at::maybe_wrap_dim(*dim_, this->dim());
  TORCH_CHECK(dim > 0, "Cannot squeeze first dimension.");
  TORCH_CHECK(
      ((sizes()[dim]) && ((*(sizes()[dim])) == 1)),
      "Given dimension is either undefined or not a singleton.");
  if (dim < this->nested_dim()) {
    _structure = _squeeze_nested_dim(_structure, dim);
  } else {
    int64_t height = _structure.height();
    _structure =
        map([dim, height](
                at::Tensor tensor) { return tensor.squeeze(dim - height); },
            _structure);
  }
  _first_variable =
      get_first_leaf(_structure) ? *get_first_leaf(_structure) : at::ones({});
  _nested_size = infer_nested_size(_structure);
  return *this;
}

} // namespace nested_tensor
} // namespace torch

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
  return wrap_nested_tensor(get_nested_tensor(self).contiguous());
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

Tensor NestedTensor_clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto self_impl = get_nested_tensor_impl(src);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([&optional_memory_format](Tensor a) {
          return at::clone(a, optional_memory_format);
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
  m.impl_UNBOXED("copy_", NestedTensor_copy_);
  m.impl_UNBOXED("squeeze_", NestedTensor_squeeze_);
  m.impl_UNBOXED("squeeze_.dim", NestedTensor_squeeze__dim);
  m.impl_UNBOXED("squeeze", NestedTensor_squeeze);
  m.impl_UNBOXED("squeeze.dim", NestedTensor_squeeze_dim);
  m.impl_UNBOXED("contiguous", NestedTensor_contiguous);
  m.impl_UNBOXED("is_pinned", NestedTensor_is_pinned);
  m.impl_UNBOXED("unbind.int", NestedTensor_unbind);
  m.impl_UNBOXED("select.int", NestedTensor_select);
}

}
