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

std::vector<c10::optional<int64_t>> NestedTensorImpl::opt_sizes() const {
  return construct_size(
      map([](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
          get_structure()));
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

SizeNode infer_nested_size(const TensorNode& _structure) {
  return map(
      [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
      _structure);
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

NestedTensorImpl::NestedTensorImpl(TensorNode structure)
    : TensorImpl(
          c10::DispatchKeySet(NestedTensorKey),
          get_first_leaf(structure) ? get_first_leaf(structure)->dtype()
                                    : at::ones({}).dtype(),
          get_first_leaf(structure) ? get_first_leaf(structure)->device()
                                    : at::ones({}).device()),
      _structure(structure),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(map(
          [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
          _structure)) {
  TORCH_CHECK(
      !_structure.is_leaf(),
      "NestedTensorImpl must be given structure of at least height 1.")
  for (auto opt_int : construct_size(_nested_size)) {
    if (opt_int) {
      _sizes.push_back(*opt_int);
    } else {
      // TODO: Should we prefer this over opt_sizes?
      // TODO: Using -1 here is of of a similar thought as using -1 in reshape
      // as a placeholder. Unfortunatly using -1 here interacts very badly with
      // the rest of the functions that consume size.
      _sizes.push_back(0);
    }
  }
}

inline TensorNode _squeeze_nested_dim(TensorNode structure, int64_t dim) {
  if (dim == 0) {
    return structure.children(0);
  }
  return TensorNode(_squeeze_nested_dim(structure, dim - 1));
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

at::Tensor NestedTensorImpl::to_tensor() {
  // TODO: Not necessarily a view because of stack and reshape.
  std::vector<int64_t> new_size;
  for (const auto& si : opt_sizes()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t at
      // this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  return _to_tensor(get_structure());
}

Tensor NestedTensorImpl::to_nested_tensor(c10::optional<int64_t> dim__) {
  int64_t dim_ = 0;
  if (dim__) {
    dim_ = *dim__;
  }
  int64_t dim = at::maybe_wrap_dim(dim_, this->dim());
  // if dim < nested_dim() the NestedTensor is already nested
  // up to the given dimension.
  if (dim >= nested_dim()) {
    TensorNode unbound = _unbind_tensors(get_structure());
    for (int64_t i = 0; i < (dim - nested_dim()); i++) {
      unbound = _unbind_tensors(unbound);
    }
    return wrap_tensor_node(std::move(unbound));
  }
  return wrap_tensor_node(std::move(_structure));
}

bool is_nested_tensor_impl(const at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

TensorNode get_nested_tensor_structure(at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    return TensorNode(std::move(tensor));
  }
  return get_nested_tensor_impl(tensor)->get_structure();
}

at::Tensor wrap_tensor_node(TensorNode&& result) {
  if (result.is_leaf()) {
    return result.payload();
  }
  return at::detail::make_tensor<NestedTensorImpl>(result);
}

std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode> input) {
  std::vector<at::Tensor> result;
  for (size_t i = 0; i < input.size(); i++) {
    result.push_back(wrap_tensor_node(std::move(input[i])));
  }
  return result;
}

int64_t NestedTensorImpl::size(int64_t dim) const {
  std::vector<c10::optional<int64_t>> size = opt_sizes();
  if (size[dim]) {
    return *(size[dim]);
  }
  throw std::runtime_error(
      "NestedTensor size at dim is not Tensor shape compliant.");
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
  return wrap_tensor_node(
      map([](at::Tensor tensor) { return tensor.contiguous(); },
          get_nested_tensor_structure(self)));
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_) {
  auto impl_data = get_nested_tensor_impl(tensor);
  if (!dim_) {
    return impl_data->to_tensor();
  }
  int64_t dim = maybe_wrap_dim((*dim_), impl_data->dim());
  if (dim == 0) {
    return impl_data->to_tensor();
  }
  // If dim is bigger than nested_dim the NestedTensor is already
  // of Tensor for dimensions bigger than the given.
  if (impl_data->nested_dim() == 1) {
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
      auto s = get_nested_tensor_impl(ci)->get_structure();
      result.push_back(TensorNode(std::move(s)));
    } else {
      // TODO: If it's a NestedTensor instance get the structure
      result.push_back(TensorNode(std::move(ci)));
    }
  }
  return wrap_tensor_node(TensorNode(std::move(result)));
}

bool NestedTensor_is_pinned(const Tensor& self) {
  return get_nested_tensor_impl(self)->is_pinned();
}

std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  auto _data = get_nested_tensor_impl(self);
  dim = at::maybe_wrap_dim(dim, _data->dim());
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
  std::cout << "HEEEE" << " -dim: " << dim << " -index " << index << std::endl;
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  if (dim != 0) {
    TORCH_CHECK_INDEX(false, "select() only supports dim == 0 for now.");
  }
  auto tmp = get_nested_tensor_structure(self).unbind()[index];
  return wrap_tensor_node(std::move(tmp));
}

Tensor NestedTensor_slice(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  if (dim != 0) {
    TORCH_CHECK_INDEX(false, "slice() only supports dim == 0 for now.");
  }
  // TODO: support negative strides
  TORCH_CHECK(step >= 1, "slice step must be positive for now.");
  int64_t sizes_0 = self.size(0);
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

Tensor NestedTensor_clone(
    const Tensor& src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return wrap_tensor_node(map(
      [&optional_memory_format](Tensor a) {
        return at::clone(a, optional_memory_format);
      },
      get_nested_tensor_structure(src)));
}

Tensor& NestedTensor_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  auto self_data = get_nested_tensor_impl(self);
  auto src_data = get_nested_tensor_impl(src);
  TORCH_CHECK(
      shape_matches(self_data->nested_size(), src_data->nested_size()),
      "self and source don't match in shape");
  apply(
      [](at::Tensor& self, at::Tensor& source) { return self.copy_(source); },
      self_data->get_structure(),
      src_data->get_structure());
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
  int64_t dim = at::maybe_wrap_dim(*dim_, self.dim());
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
  return wrap_tensor_node(map(
      [dim, height](at::Tensor tensor) { return tensor.squeeze(dim - height); },
      self_impl->get_structure()));
}

Tensor& NestedTensor_squeeze_(Tensor& self) {
  self = _NestedTensor_squeeze_(self, c10::nullopt);
  return self;
}

Tensor& NestedTensor_squeeze__dim(Tensor& self, int64_t dim) {
  self = _NestedTensor_squeeze_(self, dim);
  return self;
}

Tensor NestedTensor_squeeze(const Tensor& self) {
  auto new_tensor = NestedTensor_clone(self, c10::nullopt);
  return _NestedTensor_squeeze_(new_tensor, c10::nullopt);
}

Tensor NestedTensor_squeeze_dim(const Tensor& self, int64_t dim) {
  auto new_tensor = NestedTensor_clone(self, c10::nullopt);
  return _NestedTensor_squeeze_(new_tensor, dim);
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
  m.impl_UNBOXED("slice.Tensor", NestedTensor_slice);
}
} // namespace at
