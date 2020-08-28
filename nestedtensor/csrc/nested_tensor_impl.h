#pragma once
#include <ATen/ATen.h>
#include <c10/util/Metaprogramming.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

#define TRACEPACKED 1

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

constexpr auto NestedTensorKey_PreAutograd =
    DispatchKey::PrivateUse1_PreAutograd;
constexpr auto NestedTensorKey = DispatchKey::PrivateUse1;

struct NestedTensorImpl;

template <class A>
bool is_nested_tensor_impl(A tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey) ||
      tensor.unsafeGetTensorImpl()->key_set().has(
          at::NestedTensorKey_PreAutograd);
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
inline bool tensor_shape_matches(A a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a), "Can only compare shapes of NestedTensors.");
  return true;
}

template <class A, class B>
inline bool tensor_shape_matches(A a, B b) {
  TORCH_CHECK(
      is_nested_tensor_impl(a, b), "Can only compare shapes of NestedTensors.");
  return shape_matches(
      get_nested_tensor_structure(a), get_nested_tensor_structure(b));
}

template <class A, class B, class... C>
inline bool tensor_shape_matches(A a, B b, C... c) {
  TORCH_CHECK(
      is_nested_tensor_impl(a, b, c...),
      "Can only compare shapes of NestedTensors.");
  return shape_matches(
             get_nested_tensor_structure(a), get_nested_tensor_structure(b)) &&
      tensor_shape_matches(b, c...);
}

template <class A>
inline bool nested_size_matches(A a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a), "Can only compare shapes of NestedTensors.");
  return true;
}

template <class A, class B>
inline bool nested_size_matches(A a, B b) {
  TORCH_CHECK(
      is_nested_tensor_impl(a, b), "Can only compare shapes of NestedTensors.");
  auto nested_size_a = get_nested_tensor_impl(a)->nested_size();
  auto nested_size_b = get_nested_tensor_impl(b)->nested_size();
  if (!shape_matches(nested_size_a, nested_size_b)) {
    return false;
  }
  auto bools = map(
      [](c10::List<int64_t> a, c10::List<int64_t> b) {
        if (a.size() != b.size()) {
          return false;
        }
        for (int64_t i = 0; i < a.size(); i++) {
          if (a[i] != b[i]) {
            return false;
          }
        }
        return true;
      },
      nested_size_a,
      nested_size_b);
  return all(bools);
}

template <class A, class B, class... C>
inline bool nested_size_matches(A a, B b, C... c) {
  return nested_size_matches(a, b) && nested_size_matches(b, c...);
}

template <class... A>
inline void torch_check_tensor_shape_matches(A... a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a...), "Can only check shapes of NestedTensors.");
  TORCH_CHECK(tensor_shape_matches(a...), "NestedTensor shapes don't match.");
}

template <class A>
bool is_packed(A tensor) {
  return get_nested_tensor_structure(tensor).buffer().has_value();
}

template <class A, class B>
bool is_packed(A first, B other) {
  return is_packed(first) && is_packed(other);
}

template <class A, class B, class... C>
bool is_packed(A first, B second, C... other) {
  return is_packed(first, second) && is_packed(other...);
}

template <class F, class... A>
static inline void apply_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  apply(std::move(fn), get_nested_tensor_structure(a)...);
}

at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor);
torch::nested_tensor::TensorNode get_nested_tensor_structure(at::Tensor tensor);

at::Tensor wrap_tensor_node(NestedTensorImpl);
at::Tensor wrap_tensor_node(TensorNode&&);
std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode>);

template <class F, class... A>
static inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  return wrap_tensor_node(
      map(std::move(fn), get_nested_tensor_structure(a)...));
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
  Tensor to_nested_tensor(c10::optional<int64_t> dim);
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
  at::Tensor to_tensor();

  std::vector<c10::optional<int64_t>> opt_sizes() const;
  IntArrayRef sizes() const override {
    return IntArrayRef(_sizes);
  }
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

 private:
  TensorNode _structure;
  at::Tensor _first_variable;
  SizeNode _nested_size;
  std::vector<int64_t> _sizes;
};

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
struct _Function_no_bw {};

template <class FuncPtr, class... Parameters>
struct _Function_no_bw<FuncPtr, c10::guts::typelist::typelist<Parameters...>>
    : public torch::autograd::Function<_Function_no_bw<
          FuncPtr,
          c10::guts::typelist::typelist<Parameters...>>> {
  using ReturnType = typename c10::guts::infer_function_traits_t<
      typename FuncPtr::FuncType>::return_type;
  static ReturnType forward(
      torch::autograd::AutogradContext* ctx,
      Parameters... args) {
    return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(false, "Backward not implemented for ", typeid(FuncPtr).name());
    return {};
  }
};

template <
    class Tuple,
    class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> to_vector(Tuple&& tuple) {
  return c10::guts::apply(
      [](auto&&... elems) {
        return std::vector<T>{std::forward<decltype(elems)>(elems)...};
      },
      std::forward<Tuple>(tuple));
}

template <class FuncPtr, class ParameterTypes>
struct _Function_no_bw_wrapper {};

// you have to create a wrapper struct to create a version of apply that only
// accepts the arguments defined in forward. torch::autograd::Function::apply
// accepts any arguments regardless of what signature
// torch::autograd::Function::forward has and therefore you can't resolve it's
// signature. Instead you'd expect apply to have the exact same signature as
// forward
template <class FuncPtr, class... Parameters>
struct _Function_no_bw_wrapper<
    FuncPtr,
    c10::guts::typelist::typelist<Parameters...>> {
  using AutogradFunction =
      _Function_no_bw<FuncPtr, c10::guts::typelist::typelist<Parameters...>>;
  using ReturnType = typename c10::guts::infer_function_traits_t<
      typename FuncPtr::FuncType>::return_type;
  static ReturnType apply(Parameters... args) {
    return AutogradFunction::apply(args...);
  }
};

template <class FuncPtr>
constexpr auto no_bw(FuncPtr /*func_ptr*/) {
  using function_traits =
      c10::guts::infer_function_traits_t<typename FuncPtr::FuncType>;
  using parameter_types = typename function_traits::parameter_types;
  using AutogradFunctionWrapper =
      _Function_no_bw_wrapper<FuncPtr, parameter_types>;
  return &AutogradFunctionWrapper::apply;
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

// The approach here is quite "simple". There are six different stages to this.
// 1. We take the input NestedTensor whose constituents are, by design, required
// to not track gradients. Only the NestedTensor as a whole is allowed to track
// that information.
// 2. We take that NestedTensor and create a copy, i.e. a new NestedTensor,
// where the gradients do track gradients. This is not a valid NestedTensor
// outside the context of this function and in the future we might decide to
// pick a different container, maybe even a flat list, for this purpose.
// 3. We set these constiuents of the new NestedTensor to track gradients. A
// very important point here is that within a custom autograd Function
// AutoGradMode is *disabled*, because we're defining a new elementary operation
// within the Autograd graph and aren't appending to it. We're effectively
// creating a subgraph for the purpose of this operation here that isn't connect
// to the overall graph that corresponds to NestedTensor operations.
// 4. We apply the differentiable function that was passed as an argument to
// each constiuents of the NestedTensor from step 3 again while enabling
// AutoGradMode. We will again get a NestedTensor where the constituents track
// gradients. To make sure we actually return a valid NestedTensor we detach
// this information for our return value and save the NestedTensor from this
// step only for the backward pass.
// 5. This step does the actual detach of the constituents
// 6. This step then returns the NestedTensor from step 5.
//
// NOTE: This doesn't account for propagating gradients to gradient carrying
// functions caught in the closure of func. For example, batchnorm will want
// to get gradients for its weight and bias. If they are regular Tensors
// they won't be given as inputs and their gradients won't be propagated
// by this mapper.
template <typename F, class... A>
struct NestedTensorFunction_mapper
    : public torch::autograd::Function<NestedTensorFunction_mapper<F, A...>> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      F&& fn,
      // 1. Original NestedTensors
      A... input) {
    auto autograd_input_tuple_ =
        c10::guts::tuple_map(std::tuple<A...>(input...), [](at::Tensor t) {
          apply_nested_tensor(
              [](at::Tensor& ti) {
                TORCH_CHECK(
                    !ti.requires_grad(),
                    "autograd_mapper input's constituents shouldn't require gradients.");
              },
              t);
          // if (t.requires_grad()) {
          return map_nested_tensor(
              // 2. Constituents of NestedTensors
              [](at::Tensor ti) {
                AutoGradMode autogradmode(true);
                // TODO: Don't apply this if the corresponding NestedTensor
                // doesn't require a gradient.
                auto alias = ti.alias();
                alias.requires_grad_();
                // 3. Alias to constituents that do requires gradients
                return alias;
              },
              t);
          // }
          // return t;
        });
    auto autograd_input_tuple = autograd_input_tuple_;

    // 4. Output of differentiable function given Tensor from step 3.
    at::Tensor autograd_output = c10::guts::apply(
        [&fn](A... a) {
          return map_nested_tensor(
              [&](A... t) {
                AutoGradMode autogradmode(true);
                auto result = fn(t...);
                TORCH_CHECK(
                    result.requires_grad(),
                    "fn ",
                    typeid(F).name(),
                    " doesn't seem differentiable.");
                return result;
              },
              a...);
        },
        std::move(autograd_input_tuple_));

    auto tensor_vector = to_vector(std::move(autograd_input_tuple));
    tensor_vector.push_back(autograd_output);
    ctx->save_for_backward(tensor_vector);

    // 5. Constituents of output NestedTensor
    auto output = map_nested_tensor(
        [](at::Tensor t) { return t.alias().detach(); }, autograd_output);

    // 6. Output NestedTensor
    return output;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output_) {
    auto saved_data = ctx->get_saved_variables();
    TORCH_CHECK(
        grad_output_.size() == 1,
        "Only one incoming gradient support for now.");
    TORCH_CHECK(
        saved_data.size() == 2,
        "Only one input and one output supported for now.");
    at::Tensor grad_input;
    grad_input = map_nested_tensor(
        [](at::Tensor r, at::Tensor i, at::Tensor g) {
          auto result = torch::autograd::grad({r}, {i}, {g});
          TORCH_CHECK(result.size() == 1, "not supported 2");
          return result[0];
        },
        saved_data[1],
        saved_data[0],
        grad_output_[0]);

    at::Tensor undef;
    return {undef, grad_input};
  }
};

template <class F, class... A>
static inline at::Tensor autograd_map_nested_tensor(F&& fn, A... a) {
  return NestedTensorFunction_mapper<F, A...>::apply(std::move(fn), a...);
}

#ifdef TRACEPACKED
#define nt_impl(M, NAME, FUNC) M.impl_UNBOXED(NAME, trace(TORCH_FN(FUNC)))
#else
#define nt_impl(M, NAME, FUNC) M.impl_UNBOXED(NAME, FUNC)
#endif

} // namespace at
