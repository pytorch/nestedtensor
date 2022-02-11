#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>
#include <c10/core/DispatchKey.h>
#include <nestedtensor/csrc/transpose.h>

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

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       EfficientSizeNode nested_size,
       EfficientSizeNode nested_stride)
    : TensorImpl(
          c10::DispatchKeySet({NestedTensorKey}),
          buffer.dtype(),
          buffer.device()),
      _buffer(buffer),
      _nested_size(nested_size),
      _nested_stride(nested_stride),
      _is_pinned(_buffer.is_pinned()),
      _is_contiguous(torch::nested_tensor::impl::storage_is_contiguous(
          _buffer,
          _nested_size,
          _nested_stride)),
      _is_contiguous_channels_last(torch::nested_tensor::impl::storage_is_contiguous_channels_last(
          _buffer,
          _nested_size,
          _nested_stride)) {
  remove_autograd_key();
  key_set_ = key_set_ - c10::DispatchKeySet({c10::DispatchKey::ADInplaceOrView});
}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       EfficientSizeNode nested_size)
  : NestedTensorImpl(std::move(buffer),
                     nested_size,
                     torch::nested_tensor::impl::_cont_stride(nested_size)) {}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       SizeNode nested_size,
       SizeNode nested_stride)
  : NestedTensorImpl(std::move(buffer),
                     EfficientSizeNode(nested_size),
                     EfficientSizeNode(nested_stride)) {}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       SizeNode nested_size)
  : NestedTensorImpl(std::move(buffer),
                     EfficientSizeNode(nested_size)) {}

NestedTensorImpl::NestedTensorImpl(TensorNode structure)
  : NestedTensorImpl(
             torch::nested_tensor::impl::pack(structure),
             EfficientSizeNode(
               map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                 structure))) {}


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
  if (nested_size.is_leaf()) {
    return buffer.reshape(IntArrayRef(nested_size.payload()));
  }
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), nested_size);
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
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      efficient_nested_size,
      efficient_nested_stride);
}

at::Tensor wrap_buffer(
    at::Tensor&& buffer,
    EfficientSizeNode efficient_nested_size) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  TORCH_CHECK(
      efficient_nested_size.height() > 0,
      "Internal error: expected nested_size of non-zero height.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      efficient_nested_size);
}

Tensor NestedTensor_contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (get_is_contiguous(self, memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  if (memory_format == at::MemoryFormat::Contiguous) {
    if (get_is_contiguous(self, c10::MemoryFormat::ChannelsLast)) {
      auto transposed_sizes = map_efficient_size([](int64_t* size_ptr, int64_t size) {
          // nchw
          int64_t tmp = size_ptr[0];
          size_ptr[0] = size_ptr[2];
          size_ptr[2] = tmp;
          // nwhc
          tmp = size_ptr[0];
          size_ptr[0] = size_ptr[1];
          size_ptr[1] = tmp;
          // nhwc
          }, get_efficient_nested_size(self));
      Tensor self_transposed = wrap_buffer(get_buffer(self), transposed_sizes);
      return transpose_nhwc_nchw(self_transposed);
    }
    return at::detail::make_tensor<NestedTensorImpl>(get_nested_tensor_structure(self));
  }
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    Tensor self_cont = self;
    if (!get_is_contiguous(self, c10::MemoryFormat::Contiguous)) {
      self_cont = NestedTensor_contiguous(self, at::MemoryFormat::Contiguous);
    }
    TORCH_CHECK(get_dim(self_cont) == 4, "ChannelsLast memory format requires 4 dim input.");
    auto new_strides = map_efficient_size([](int64_t* stride_ptr, int64_t* size_ptr, int64_t size) {
        stride_ptr[2] = size_ptr[0];
        stride_ptr[1] = stride_ptr[2] * size_ptr[2];
        stride_ptr[0] = 1;
        }, get_efficient_nested_stride(self_cont), get_efficient_nested_size(self_cont));
    self_cont = transpose_nchw_nhwc(self_cont);
    return wrap_buffer(get_buffer(self_cont), get_efficient_nested_size(self), new_strides);
  }
  TORCH_CHECK(false, "Given memory format ", memory_format, " not supported by NestedTensor_contiguous.");
  return self;
}

bool NestedTensor_is_pinned(const Tensor& self, c10::optional<Device> device) {
  TORCH_CHECK(
      !device.has_value() || device->is_cuda(),
      "NestedTensor doesn't support non-CUDA pinned memory");
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

Tensor NestedTensor_to_dtype_layout(
  const Tensor& self,
  c10::optional<ScalarType> dtype,
  c10::optional<Layout> layout,
  c10::optional<Device> device,
  c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format) {
    auto input_buffer = get_buffer(self);
    auto result_nt = wrap_buffer(input_buffer.to(dtype, layout, device, pin_memory,
                                                 non_blocking, copy, c10::nullopt),
                                 get_efficient_nested_size(self),
                                 get_efficient_nested_stride(self));
    if (optional_memory_format) {
      return NestedTensor_contiguous(result_nt, *optional_memory_format);
    }
    return result_nt;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "contiguous", NestedTensor_contiguous);
  nt_impl(m, "copy_", NestedTensor_copy_);
  nt_impl(m, "is_pinned", NestedTensor_is_pinned);
  nt_impl(m, "select.int", NestedTensor_select);
  nt_impl(m, "size.int", NestedTensor_size_int);
  nt_impl(m, "slice.Tensor", NestedTensor_slice);
  nt_impl(m, "squeeze", NestedTensor_squeeze);
  nt_impl(m, "squeeze.dim", NestedTensor_squeeze_dim);
  nt_impl(m, "squeeze_", NestedTensor_squeeze_);
  nt_impl(m, "squeeze_.dim", NestedTensor_squeeze__dim);
  nt_impl(m, "unbind.int", NestedTensor_unbind);
  nt_impl(m, "unsqueeze", NestedTensor_unsqueeze);
  nt_impl(m, "to.dtype_layout", NestedTensor_to_dtype_layout);
}
} // namespace at
