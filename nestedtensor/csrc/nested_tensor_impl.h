#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

// #define TRACEPACKED 1
// #define USEPACKED 1

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

} // namespace nested_tensor
} // namespace torch

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

template <class A>
void torch_check_is_nested_tensor(A tensor) {
  TORCH_CHECK(is_nested_tensor_impl(tensor), "Argument is not NestedTensor.");
}

template <class A, class B>
void torch_check_is_nested_tensor(A first, B other) {
  torch_check_is_nested_tensor(first);
  torch_check_is_nested_tensor(other);
}

template <class A, class B, class... C>
void torch_check_is_nested_tensor(A first, B second, C... other) {
  torch_check_is_nested_tensor(first, second);
  torch_check_is_nested_tensor(other...);
}

template <class A>
inline bool tensor_shape_matches(A a) {
  return true;
}

template <class A, class B>
inline bool tensor_shape_matches(A a, B b) {
  if (is_nested_tensor_impl(a, b)) {
    return shape_matches(
        get_nested_tensor_structure(a), get_nested_tensor_structure(b));
  }
  return true;
}

template <class A, class B, class... C>
inline bool tensor_shape_matches(A a, B b, C... c) {
  TORCH_CHECK(
      is_nested_tensor_impl(a, b, c...),
      "Can only compare shapes of NestedTensors.");
  if (is_nested_tensor_impl(a, b)) {
    return shape_matches(
               get_nested_tensor_structure(a),
               get_nested_tensor_structure(b)) &&
        tensor_shape_matches(b, c...);
  }
  if (is_nested_tensor_impl(a)) {
    return tensor_shape_matches(a, c...);
  }
  if (is_nested_tensor_impl(b)) {
    return tensor_shape_matches(b, c...);
  }
  return tensor_shape_matches(c...);
}

template <class A>
inline bool nested_size_matches(SizeNode a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a), "Can only compare shapes of NestedTensors.");
  return true;
}

template <class A, class B>
inline bool nested_size_matches(A nested_size_a, B nested_size_b) {
  if (!shape_matches(nested_size_a, nested_size_b)) {
    return false;
  }
  std::vector<bool> bools = flatten(map(
      [](c10::List<int64_t> a, c10::List<int64_t> b) -> bool {
        if (a.size() != b.size()) {
          return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
          if (a[i] != b[i]) {
            return false;
          }
        }
        return true;
      },
      nested_size_a,
      nested_size_b));
  bool all = true;
  for (size_t i = 0; i < bools.size(); i++) {
    all = all && bools[i];
  }
  return all;
}

template <class A, class B, class... C>
inline bool nested_size_matches(A a, B b, C... c) {
  return nested_size_matches(a, b) && nested_size_matches(b, c...);
}

template <class... A>
inline void torch_check_tensor_shape_matches(A... a) {
  TORCH_CHECK(tensor_shape_matches(a...), "NestedTensor shapes don't match.");
}

template <class F, class... A>
static inline void apply_nested_tensor(F&& fn, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  apply(std::move(fn), get_nested_tensor_structure(a)...);
}

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(TensorNode structure);

  int64_t dim() const override {
    return _first_variable.dim() + nested_dim();
  }
  int64_t numel() const override {
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + leaf.numel();
    };
    return reduce<decltype(fn), int64_t, at::Tensor>(get_structure(), fn, 0);
  }
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    // NOTE: The Tensors themselves might not be contiguous even if there is a
    // buffer. For this to be contiguous not only the individuals Tensors have
    // to be but also the buffer.
    auto fn = [](at::Tensor leaf, bool input) {
      return input && leaf.is_contiguous();
    };
    return reduce<decltype(fn), bool, at::Tensor>(get_structure(), fn, true) &&
        get_structure().buffer().has_value();
  }
  TensorNode& get_structure() {
    return _structure;
  }
  const TensorNode& get_structure() const {
    return _structure;
  }
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  // TODO:
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
  int64_t nested_dim() const {
    return get_structure().height();
  }
  bool is_pinned() const {
    return _first_variable.is_pinned();
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
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
        get_structure());
  }
  SizeNode nested_stride() const {
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.strides()); },
        get_structure());
  }

  std::vector<c10::optional<int64_t>> opt_sizes() const;
  IntArrayRef sizes() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
    return IntArrayRef(_sizes);
  }
  IntArrayRef strides() const override;

 private:
  TensorNode _structure;
  at::Tensor _first_variable;
  SizeNode _nested_size;
  std::vector<int64_t> _sizes;
};

int64_t nt_size(Tensor tensor, int64_t dim);

Tensor NestedTensor_to_nested_tensor(
    at::Tensor input,
    c10::optional<int64_t> dim__);

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node);

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

template <class A>
static inline bool is_packed(A tensor) {
  return is_nested_tensor_impl(tensor) &&
      get_nested_tensor_structure(tensor).buffer().has_value();
}

template <class A, class B>
static inline bool is_packed(A first, B other) {
  return is_packed(first) && is_packed(other);
}

template <class A, class B, class... C>
static inline bool is_packed(A first, B second, C... other) {
  return is_packed(first, second) && is_packed(other...);
}

static inline at::Tensor get_buffer(const at::Tensor& tensor) {
  TORCH_CHECK(is_packed(tensor), "Given Tensor doesn't have buffer.");
  return *(get_nested_tensor_structure(tensor).buffer());
}

static inline std::vector<c10::optional<int64_t>> get_opt_sizes(
    at::Tensor tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->opt_sizes();
}

static inline SizeNode get_nested_size(at::Tensor tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->nested_size();
}

static inline int64_t get_nested_dim(const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->nested_dim();
}

at::Tensor wrap_tensor_node(NestedTensorImpl);
at::Tensor wrap_tensor_node(TensorNode&&);
std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode>);
at::Tensor wrap_buffer(at::Tensor&&, SizeNode nested_size);

template <class F, class... A>
static inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  // torch_check_tensor_shape_matches(a...);
  // torch_check_is_nested_tensor(a...);
  return wrap_tensor_node(
      map(std::move(fn), get_nested_tensor_structure(a)...));
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
// #define nt_impl(M, NAME, FUNC) M.impl_UNBOXED(NAME, trace(TORCH_FN(FUNC)))
#define nt_impl(M, NAME, FUNC) \
  M.impl(                      \
      NAME,                    \
      torch::CppFunction::makeFromUnboxedFunction(trace(TORCH_FN(FUNC))))
#else
// #define nt_impl(M, NAME, FUNC) M.impl_UNBOXED(NAME, FUNC)
#define nt_impl(M, NAME, FUNC) M.impl(NAME, TORCH_FN(FUNC))
#endif

} // namespace at
