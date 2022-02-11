#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <nestedtensor/csrc/storage/Packed.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

// #define TRACEPACKED 1
// #define USEPACKED 1

namespace at {

using namespace torch::nested_tensor;

constexpr auto NestedTensorKey = DispatchKey::NestedTensor;

struct NestedTensorImpl;

template <class A>
bool is_nested_tensor_impl(A tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

template <class A, class B>
bool is_nested_tensor_impl(A first, B other) {
  return is_nested_tensor_impl(first) && is_nested_tensor_impl(other);
}

template <class A, class B, class... C>
bool is_nested_tensor_impl(A first, B second, C... other) {
  return is_nested_tensor_impl(first, second) &&
      is_nested_tensor_impl(other...);
}

template <class F, class... A>
inline void apply_nested_tensor(F&& fn, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  apply(std::forward<F>(fn), get_nested_tensor_structure(a)...);
}

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(at::Tensor&& buffer, EfficientSizeNode nested_size, EfficientSizeNode nested_stride);
  explicit NestedTensorImpl(at::Tensor&& buffer, EfficientSizeNode nested_size);
  explicit NestedTensorImpl(at::Tensor&& buffer, SizeNode nested_size, SizeNode nested_stride);
  explicit NestedTensorImpl(at::Tensor&& buffer, SizeNode nested_size);
  explicit NestedTensorImpl(TensorNode structure);

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  int64_t dim() const override {
    TORCH_CHECK(
        false, "dim is disabled. These methods are not virtual in fbcode.");
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  int64_t numel() const override {
    TORCH_CHECK(
        false, "numel is disabled. These methods are not virtual in fbcode.");
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    TORCH_CHECK(
        false,
        "is_contiguous is disabled. These methods are not virtual in fbcode.");
  }
#endif
  TensorNode get_structure() const {
    return std::get<0>(torch::nested_tensor::impl::build_structure(
        _buffer.reshape({-1}),
        _nested_size,
        _nested_stride));
  }
  EfficientSizeNode get_nested_size() {
    return _nested_size;
  }
  EfficientSizeNode get_nested_stride() {
    return _nested_stride;
  }
  int64_t nested_dim() const {
    return _nested_size.height();
  }
  bool is_pinned() const {
    return _buffer.is_pinned();
  }
  // This is a C++ representation of a nested list of torch.Sizes
  //
  // It can never be a list of just numbers, because torch.Size
  // is always a list and NestedTensors represent lists of torch.Tensors
  //
  // Noteworthy cases:
  //
  // This is an empty list of lists if we construct
  // nested_tensor([])
  // which is of nested_dim 1, dim 1 and tensor_dim 0
  //
  // This is a list of empty lists if we construct e.g.
  // nested_tensor([torch.tensor(0), torch.tensor(1), ...])
  // which is of nested_dim 1, dim 1 and tensor_dim 0
  //
  // This is a list of list of numbers if we construct e.g.
  // nested_tensor([torch.tensor([1]), torch.tensor([2]), ...])
  // which is of nested_dim 1, dim 2 and tensor_dim 1
  //
  // That means, if the list is not empty it is either a list of
  // lists of numbers or a list of empty lists.
  SizeNode nested_size() const {
    return _nested_size.to_size_node();
  }
  SizeNode nested_stride() const {
    return _nested_stride.to_size_node();
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    return _nested_size.opt_sizes();
  }
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  IntArrayRef sizes() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
    std::vector<int64_t> sizes;
    return IntArrayRef(sizes);
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  IntArrayRef strides() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
    std::vector<int64_t> strides;
    return IntArrayRef(strides);
  }
#endif

  const at::Tensor& get_buffer() const {
    return _buffer;
  }

  at::Tensor& get_buffer() {
    return _buffer;
  }

  bool get_is_cuda() const {
    return _buffer.is_cuda();
  }

  bool get_is_contiguous(at::MemoryFormat memory_format) const {
    if (memory_format == at::MemoryFormat::Contiguous) {
      return _is_contiguous;
    }
    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return _is_contiguous_channels_last;
    }
    TORCH_CHECK(false, "is_contiguous does not support memory format ", memory_format);
    return false;
  }

  bool get_is_pinned() const {
    return _is_pinned;
  }

 private:
  at::Tensor _buffer;
  const EfficientSizeNode _nested_size;
  const EfficientSizeNode _nested_stride;
  bool _is_pinned;
  const bool _is_contiguous;
  const bool _is_contiguous_channels_last;
};

int64_t nt_size(Tensor tensor, int64_t dim);

Tensor NestedTensor_to_nested_tensor(
    at::Tensor input,
    c10::optional<int64_t> dim__);

inline at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

template <class A>
inline NestedNode<A> get_nested_tensor_structure(A tensor) {
  return NestedNode<A>(std::move(tensor));
}

template <>
inline TensorNode get_nested_tensor_structure(at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    return TensorNode(std::move(tensor));
  }
  return get_nested_tensor_impl(tensor)->get_structure();
}

inline at::Tensor get_buffer(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_buffer();
}

inline const std::vector<c10::optional<int64_t>> get_opt_sizes(
    const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->opt_sizes();
}

inline const EfficientSizeNode get_efficient_nested_size(const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->get_nested_size();
}

inline const EfficientSizeNode get_efficient_nested_stride(const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->get_nested_stride();
}

inline SizeNode get_nested_size(at::Tensor tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->get_nested_size().to_size_node();
}

inline SizeNode get_nested_stride(at::Tensor tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->get_nested_stride().to_size_node();
}

inline int64_t get_dim(const at::Tensor& tensor) {
  if (is_nested_tensor_impl(tensor)) {
    return get_nested_tensor_impl(tensor)->get_nested_size().dim();
  }
  return tensor.dim();
}

inline const caffe2::TypeMeta get_dtype(const at::Tensor& tensor) {
  return tensor.dtype();
}

inline int64_t get_numel(const at::Tensor& tensor) {
  if (is_nested_tensor_impl(tensor)) {
    return get_nested_tensor_impl(tensor)->get_nested_size().numel();
  }
  return tensor.numel();
}

Tensor NestedTensor_contiguous(
    const Tensor& self,
    MemoryFormat memory_format = MemoryFormat::Contiguous);

inline bool get_is_contiguous(
    const at::Tensor& tensor,
    at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
  if (is_nested_tensor_impl(tensor)) {
    return get_nested_tensor_impl(tensor)->get_is_contiguous(memory_format);
  }
  return tensor.is_contiguous(memory_format);
}

inline bool get_is_cuda(
    const at::Tensor& tensor,
    at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
  if (is_nested_tensor_impl(tensor)) {
    return get_nested_tensor_impl(tensor)->get_is_cuda();
  }
  return tensor.is_cuda();
}

inline int64_t get_nested_dim(const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->nested_dim();
}

at::Tensor wrap_tensor_node(NestedTensorImpl);
at::Tensor wrap_tensor_node(TensorNode&&);
std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode>);
at::Tensor wrap_buffer(at::Tensor&&, SizeNode nested_size);
at::Tensor wrap_buffer(
    at::Tensor&&,
    EfficientSizeNode efficient_nested_size,
    EfficientSizeNode efficient_nested_stride);
at::Tensor wrap_buffer(
    at::Tensor&&,
    EfficientSizeNode efficient_nested_size);

template <class F, class... A>
inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  return wrap_tensor_node(
      map(std::forward<F>(fn), get_nested_tensor_structure(a)...));
}

template <class F, class I, class... A>
inline typename c10::guts::infer_function_traits<F>::type::return_type
reduce_nested_tensor(F&& fn, I init, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  return reduce(std::forward<F>(fn), init, get_nested_tensor_structure(a)...);
}

inline std::vector<at::Tensor> flatten_nested_tensor(at::Tensor tensor) {
  return flatten(get_nested_tensor_structure(tensor));
}

inline bool is_tensor_shape(const at::Tensor tensor) {
  auto nt = get_nested_tensor_impl(tensor);
  for (const auto& size : nt->opt_sizes()) {
    if (!size) {
      return false;
    }
  }
  return true;
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_);

inline Tensor NestedTensor_to_sparse_csr(Tensor tensor) {
  TORCH_CHECK(
      get_dim(tensor) == 2,
      "Given tensor must be of dimension 2, got dimension ",
      get_dim(tensor));
  Tensor values;
  if (get_is_contiguous(tensor)) {
    values = get_buffer(tensor).reshape({-1});
  } else {
    values = at::cat(flatten(get_nested_tensor_structure(tensor)));
  }
  auto tensor_sizes = get_efficient_nested_size(tensor).sizes();
  tensor_sizes = tensor_sizes.reshape({-1});
  int64_t* tensor_sizes_ptr = tensor_sizes.data_ptr<int64_t>();
  at::Tensor crow_indices =
      at::cat({torch::tensor({0}), at::cumsum(tensor_sizes, 0)});
  std::vector<at::Tensor> col_indices_;
  for (int64_t i = 0; i < tensor_sizes.size(0); i++) {
    col_indices_.push_back(torch::arange({tensor_sizes_ptr[i]}));
  }
  at::Tensor col_indices = at::cat(col_indices_);
  return at::native::sparse_csr_tensor(
      crow_indices, col_indices, values, c10::nullopt, torch::kSparseCsr);
}

inline std::ostream& operator<<(
    std::ostream& out,
    const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

template <class FuncPtr, class ParameterTypes>
struct _Function_trace_wrapper {};

template <class FuncPtr, class... Parameters>
struct _Function_trace_wrapper<
    FuncPtr,
    c10::guts::typelist::typelist<Parameters...>> {
  using ReturnType = typename c10::guts::infer_function_traits_t<
      typename FuncPtr::FuncType>::return_type;
  static ReturnType apply(Parameters... args) {
    std::cout << "Calling " << typeid(FuncPtr).name() << std::endl;
    return (*FuncPtr::func_ptr())(args...);
  }
};

template <class FuncPtr>
constexpr auto trace(FuncPtr /*func_ptr*/) {
  using function_traits =
      c10::guts::infer_function_traits_t<typename FuncPtr::FuncType>;
  using parameter_types = typename function_traits::parameter_types;
  return &_Function_trace_wrapper<FuncPtr, parameter_types>::apply;
}

#ifdef TRACEPACKED
// #define nt_impl(M, NAME, FUNC) M.impl(NAME, trace(TORCH_FN(FUNC)))
#else
// #define nt_impl(M, NAME, FUNC) M.impl(NAME, trace(TORCH_FN(FUNC)))
#define nt_impl(M, NAME, FUNC) M.impl(NAME, TORCH_FN(FUNC))
#endif

} // namespace at
