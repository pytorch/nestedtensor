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

c10::intrusive_ptr<c10::TensorImpl> NestedTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<NestedTensorImpl>(_structure);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

void NestedTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  NestedTensorImpl* nested_impl = dynamic_cast<NestedTensorImpl*>(impl.get());
  copy_tensor_metadata(
      /*src_impl=*/nested_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  nested_impl->_structure = _structure;
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
          c10::DispatchKeySet({NestedTensorKey_PreAutograd, NestedTensorKey}),
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
  return squeeze(structure, dim);
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
  return _sizes;
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

at::Tensor wrap_buffer(at::Tensor&& buffer, SizeNode nested_size) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
      std::move(buffer), nested_size));
}

struct NestedTensorFunction_contiguous
    : public torch::autograd::Function<NestedTensorFunction_contiguous> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input) {
    return wrap_tensor_node(pack(get_nested_tensor_structure(input)));
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(grad_output_.size() == 1, "grad_output must be of size 1.");
    at::Tensor grad_output = grad_output_[0];
    return {grad_output};
  }
};

Tensor NestedTensor_contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  return NestedTensorFunction_contiguous::apply(self);
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
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  if (dim != 0) {
    TORCH_CHECK_INDEX(false, "select() only supports dim == 0 for now.");
  }
  auto tmp = get_nested_tensor_structure(self).unbind()[index];
  return wrap_tensor_node(std::move(tmp));
}

Tensor NestedTensorImpl::to_nested_tensor(c10::optional<int64_t> dim__) {
  int64_t dim_ = 0;
  if (dim__) {
    dim_ = *dim__;
  }
  int64_t dim = at::maybe_wrap_dim(dim_, this->dim());
  // if dim < nested_dim() the NestedTensor is already nested
  // up to the given dimension.
  if (dim >= this->nested_dim()) {
    TensorNode unbound = _unbind_tensors(this->get_structure());
    for (int64_t i = 0; i < (dim - nested_dim()); i++) {
      unbound = _unbind_tensors(unbound);
    }
    return wrap_tensor_node(std::move(unbound));
  }
  return wrap_tensor_node(std::move(_structure));
}

// TODO: There are unanswered questions
// around 0-numel NestedTensors as maybe brought about by
// t[:, out_of_bounds:, :]
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

Tensor& NestedTensor_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  // auto self_data = get_nested_tensor_impl(self);
  // auto src_data = get_nested_tensor_impl(src);
  // TORCH_CHECK(
  //     shape_matches(self_data->nested_size(), src_data->nested_size()),
  //     "self and source don't match in shape");
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
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto self_impl = get_nested_tensor_impl(self);
  int64_t nested_dim = self_impl->nested_dim();
  TORCH_CHECK(dim > 0, "Cannot squeeze first dimension.");
  TORCH_CHECK(dim >= nested_dim, "Cannot squeeze nested dimension.");
  TORCH_CHECK(
      ((self_impl->opt_sizes()[dim]) &&
       ((*(self_impl->opt_sizes()[dim])) == 1)),
      "Given dimension is either undefined or not a singleton.");
  return autograd_map_nested_tensor(
      [dim, nested_dim](at::Tensor tensor) {
        return tensor.squeeze(dim - nested_dim);
      },
      self);
}

Tensor NestedTensor_squeeze(const Tensor& self) {
  TORCH_CHECK(false, "squeeze(Tensor) is currently not implemented.");
}

Tensor NestedTensor_unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
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

Tensor NestedTensor_as_strided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<int64_t> storage_offset_) {
  throw std::runtime_error(
      "as_strided is not implemented for NestedTensor. "
      "Please create an issue on https://github.com/pytorch/nestedtensor with your usecase.");
  return self;
}

Tensor& NestedTensor_as_strided_(
    Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<int64_t> storage_offset_) {
  throw std::runtime_error(
      "as_strided_ is not implemented for NestedTensor. "
      "Please create an issue on https://github.com/pytorch/nestedtensor with your usecase.");
  return self;
}

Tensor NestedTensor_serialize_nested_size(const Tensor& tensor) {
  auto nt_impl = get_nested_tensor_impl(tensor);
  std::vector<int64_t> out;
  torch::nested_tensor::serialize(nt_impl->nested_size(), out);
  return torch::tensor(out);
}

Tensor NestedTensor_expand_as(const Tensor& self_, const Tensor& other) {
  std::cout << "JDJDJD" << std::endl;
  at::Tensor self = self_;
  if (is_nested_tensor_impl(self, other)) {
    TORCH_CHECK(
        get_nested_tensor_impl(self)->nested_dim(),
        get_nested_tensor_impl(other)->nested_dim(),
        "Given NestedTensors need to have same nested dimension.");
    return map_nested_tensor(
        [](at::Tensor s, at::Tensor o) { return at::native::expand_as(s, o); },
        self,
        other);
  }
  TORCH_CHECK(
      !is_nested_tensor_impl(self),
      "Cannot expand a NestedTensor as a Tensor.");
  TORCH_CHECK(
      self.dim() <= other.dim(),
      "Cannot expand to a Tensor of smaller dimension.");
  while (self.dim() > 0 && self.size(0) == 1) {
    self = self.squeeze(0);
  }
  return map_nested_tensor(
      [](at::Tensor s, at::Tensor o) { return s.expand_as(o); }, self, other);
}

bool NestedTensor_sizes_equal_nt_other(
    const Tensor& self,
    IntArrayRef nested_size_other) {
  auto tmp =
      torch::nested_tensor::deserialize_size_node(nested_size_other.vec(), 0);
  SizeNode nested_size = std::get<1>(tmp);
  if (is_nested_tensor_impl(self)) {
    // std::cout << "SE1" << std::endl;
    return false;
    // return torch::nested_tensor::shape_matches(
    //     get_nested_tensor_impl(self)->nested_size(), nested_size);
  }
  // std::cout << "SE2" << std::endl;
  return false;
}

// Can nested_size_other be expanded to match the shape of grad?
// If this is true, a call to sum_to_nt will follow next in autograd/engine.cpp
bool NestedTensor_native_is_expandable_to_nt_other(
    const Tensor& self,
    IntArrayRef nested_size_other) {
  auto tmp =
      torch::nested_tensor::deserialize_size_node(nested_size_other.vec(), 0);
  SizeNode nested_size = std::get<1>(tmp);
  if (is_nested_tensor_impl(self)) {
    std::cout << "NTNE1" << std::endl;
    return false;
    // return torch::nested_tensor::shape_matches(
    //     get_nested_tensor_impl(self)->nested_size(), nested_size);
  }
  std::cout << "NTNE2" << std::endl;
  return false;
}

bool NestedTensor_native_is_expandable_to(
    const Tensor& grad,
    IntArrayRef metadata_shape) {
  std::cout << "2830283" << std::endl;
  return true;
  // return at::is_expandable_to(metadata_shape, grad.sizes());
}

void traceFallbackPre(const c10::OperatorHandle& op, Stack* stack) {
  std::cerr << "Calling autograd fallback for " << op.schema() << std::endl;
  c10::impl::ExcludeDispatchKeyGuard guard(
      c10::DispatchKey::AutogradNestedTensor);
  op.callBoxed(stack);
}

TORCH_LIBRARY_IMPL(_, AutogradNestedTensor, m) {
  // m.fallback(torch::CppFunction::makeFromBoxedFunction<&traceFallbackPre>());
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutogradNestedTensor, m) {
  nt_impl(m, "copy_", NestedTensor_copy_);
  nt_impl(m, "squeeze_", NestedTensor_squeeze_);
  nt_impl(m, "squeeze_.dim", NestedTensor_squeeze__dim);
  nt_impl(m, "squeeze", NestedTensor_squeeze);
  nt_impl(m, "squeeze.dim", NestedTensor_squeeze_dim);
  nt_impl(m, "contiguous", NestedTensor_contiguous);
  nt_impl(m, "is_pinned", NestedTensor_is_pinned);
  // nt_impl("unbind.int", no_bw(TORCH_FN(NestedTensor_unbind)));
}
TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "expand_as", NestedTensor_expand_as);
  nt_impl(m, "as_strided", NestedTensor_as_strided);
  nt_impl(m, "as_strided_", NestedTensor_as_strided_);
  nt_impl(m, "unbind.int", NestedTensor_unbind);
  nt_impl(m, "select.int", NestedTensor_select);
  nt_impl(m, "slice.Tensor", NestedTensor_slice);
  nt_impl(m, "unsqueeze", NestedTensor_unsqueeze);
  nt_impl(m, "serialize_nested_size", NestedTensor_serialize_nested_size);
  nt_impl(m, "native_is_expandable_to", NestedTensor_native_is_expandable_to);
}
// TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
//   nt_impl(m, "sizes_equal_nt_other", NestedTensor_sizes_equal_nt_other);
//   nt_impl(
//       m,
//       "native_is_expandable_to_nt_other",
//       NestedTensor_native_is_expandable_to_nt_other);
// }
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  nt_impl(m, "sizes_equal_nt_other", NestedTensor_sizes_equal_nt_other);
  nt_impl(
      m,
      "native_is_expandable_to_nt_other",
      NestedTensor_native_is_expandable_to_nt_other);
}
} // namespace at
