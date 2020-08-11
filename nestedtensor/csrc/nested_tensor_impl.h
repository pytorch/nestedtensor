#pragma once
#include <ATen/ATen.h>
#include <c10/util/Metaprogramming.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

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
  // TORCH_CHECK(
  //     get_nested_tensor_impl(a)->nested_dim() !=
  //         get_nested_tensor_impl(b)->nested_dim(),
  //     "NestedTensor nested dimensions do not match.");
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

template <class... A>
inline void torch_check_tensor_shape_matches(A... a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a...), "Can only check shapes of NestedTensors.");
  TORCH_CHECK(tensor_shape_matches(a...), "NestedTensor shapes don't match.");
}

template <class F, class... A>
static inline void apply_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  apply(std::move(fn), get_nested_tensor_structure(a)...);
}

at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor);
torch::nested_tensor::TensorNode get_nested_tensor_structure(
    const at::Tensor tensor);

at::Tensor wrap_tensor_node(NestedTensorImpl);
at::Tensor wrap_tensor_node(TensorNode&&);
std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode>);

static inline c10::optional<Tensor> undefined_to_optional(
    const at::Tensor& tensor) {
  if (tensor.defined()) {
    return c10::optional<Tensor>(tensor);
  }
  return c10::nullopt;
}

static inline Tensor optional_to_undefined(
    const c10::optional<at::Tensor>& tensor) {
  if (tensor) {
    return *tensor;
  }
  Tensor undef;
  return undef;
}

template <class F>
void apply_method(F&& fn) {}

template <class F, class A>
void apply_method(F&& fn, A a) {
  fn(a);
}

template <class F, class A, class... B>
void apply_method(F&& fn, A a, B... b) {
  fn(a);
  apply_method(fn, b...);
}

template <size_t... indices>
auto to_tuple(std::index_sequence<indices...>) {
  return std::make_tuple(indices...);
}

template <typename T>
std::vector<T> to_vector() {
  return std::vector<T>();
}

template <typename T, typename Args>
std::vector<T> to_vector(std::tuple<Args> args) {
  auto result = to_vector<T>();
  result.push_back(std::get<0>(args));
  return result;
}

template <typename T, typename Args0, typename Args1>
std::vector<T> to_vector(std::tuple<Args0, Args1> args) {
  auto result = to_vector<T, Args0>(std::make_tuple(std::get<0>(args)));
  result.push_back(std::get<1>(args));
  return result;
}

// template <typename T, typename... Args>
// std::vector<T> to_vector(std::tuple<Args...> args) {
//   auto result = to_vector<T>(c10::guts::tuple_take<
//                           decltype(args),
//                           std::tuple_size<decltype(args)>::value - 1>(args));
//   result.push_back(std::get<std::tuple_size<decltype(args)>::value -
//   1>(args)); return result;
// }

template <class F, class... A>
static inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  return wrap_tensor_node(
      map(std::move(fn), get_nested_tensor_structure(a)...));
}

// TODO: Turn this into a generic wrapper for all other operations
template <typename F, class... A>
struct NestedTensorFunction_batch_norm
    : public torch::autograd::Function<
          NestedTensorFunction_batch_norm<F, A...>> {
  static Tensor forward(torch::autograd::AutogradContext* ctx, F&& fn, A... a) {
    auto result = map_nested_tensor(
        [&](A... t) {
          AutoGradMode autogradmode(true);
          apply_method([](at::Tensor t) { t.requires_grad_(); }, t...);
          // t.requires_grad_()...;
          return fn(t...);
        },
        a...);
    ctx->save_for_backward({a..., result});
    return result;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(grad_output_.size() == 1, "Unexpected number of grad outputs.");
    auto saved = ctx->get_saved_variables();
    at::Tensor result = saved[saved.size() - 1];
    auto input_grad_output = c10::guts::tuple_map(
        to_tuple(std::index_sequence_for<A...>{}),
        [&](int64_t i) { return std::make_tuple(saved[i], grad_output_[i]); });
    auto grad_input_ = c10::guts::tuple_map(
        std::move(input_grad_output),
        [&](std::tuple<at::Tensor, at::Tensor>&& input_grad_output) {
          return map_nested_tensor(
              [](at::Tensor r, at::Tensor i, at::Tensor g) {
                return torch::autograd::grad({r}, {i}, {g})[0];
              },
              result,
              std::get<0>(input_grad_output),
              std::get<1>(input_grad_output));
        });
    torch::autograd::variable_list grad_input =
        to_vector<at::Tensor>(grad_input_);
    at::Tensor undef;
    grad_input.insert(grad_input.begin(), undef);
    return grad_input;
  }
};

template <class F, class... A>
static inline at::Tensor autograd_map_nested_tensor(F&& fn, A... a) {
  return NestedTensorFunction_batch_norm<F, A...>::apply(std::move(fn), a...);
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
    return reduce<decltype(fn), bool, at::Tensor>(get_structure(), fn, true);
  }
  TensorNode& get_structure() {
    return _structure;
  }
  const TensorNode& get_structure() const {
    return _structure;
  }
  void backward(Tensor gradient, bool retain_graph, bool create_graph) {
    apply(
        [retain_graph, create_graph](at::Tensor tensor1, at::Tensor tensor2)
            -> void { tensor1.backward(tensor2, retain_graph, create_graph); },
        get_structure(),
        get_nested_tensor_impl(gradient)->get_structure());
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
  Tensor grad() {
    auto fn = [](at::Tensor leaf, bool input) {
      return input && leaf.grad().defined();
    };
    if (!reduce<decltype(fn), bool, at::Tensor>(get_structure(), fn, true)) {
      throw std::runtime_error("Grad is undefined");
    }
    return wrap_tensor_node(
        map([](at::Tensor tensor) { return tensor.grad(); }, get_structure()));
  }
  // Tensor requires_grad_(bool requires_grad) {
  //   std::cout << "DHDHDHD" << std::endl;
  //   apply(
  //       [requires_grad](at::Tensor& tensor) -> void {
  //         tensor.set_requires_grad(requires_grad);
  //       },
  //       get_structure());
  //   return at::detail::make_tensor<NestedTensorImpl>(_structure);
  // }
  // bool requires_grad() const {
  //   std::cout << "DllrallralDHDHD" << std::endl;
  //   return _first_variable.requires_grad();
  // }
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
    // TODO: Disable this or add a check that makes sure it's Tensor shape
    // compliant and act as a convenience function.
    return IntArrayRef(_sizes);
  }
  // TODO: Throw an error if requested dimension is undefined
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

 private:
  TensorNode _structure;
  at::Tensor _first_variable;
  SizeNode _nested_size;
  std::vector<int64_t> _sizes;
};

template <class A>
inline void save_args(
    ska::flat_hash_map<std::string, at::IValue>& saved_data,
    A a) {
  saved_data["0"] = a;
}

template <class A, class... Args>
inline void save_args(
    ska::flat_hash_map<std::string, at::IValue>& saved_data,
    A a,
    Args... args) {
  saved_data[std::to_string(sizeof...(Args))] = a;
  save_args(saved_data, args...);
}

c10::IValue inline load_arg(
    ska::flat_hash_map<std::string, at::IValue>& saved_data,
    size_t i) {
  return saved_data[std::to_string(i)];
}

// TODO: Finish off this conversion
template <class Bw, Bw bw, class... Args, size_t... I>
inline std::vector<at::Tensor> load_args(
    std::vector<at::Tensor> grad_output_,
    ska::flat_hash_map<std::string, at::IValue>& saved_data,
    std::index_sequence<I...>) {
  return bw(
      load_arg(saved_data, I)
          .to<std::tuple_element_t<I, std::tuple<Args...>>>()...,
      grad_output_);
}

// TODO: Turn this into a generic wrapper for all other operations
// TODO: Needs AutoGradMode etc.
template <class Fw, Fw fw, class Bw, Bw bw, class... Args>
struct NestedTensorFunction_tie
    : public torch::autograd::Function<
          NestedTensorFunction_tie<Fw, fw, Bw, bw, Args...>> {
  static Tensor forward(torch::autograd::AutogradContext* ctx, Args... args) {
    save_args(ctx->saved_data, args...);
    return fw(args...);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(grad_output_.size() == 1, "Unexpected number of grad outputs.");
    return load_args<Bw, bw, Args...>(
        grad_output_, ctx->saved_data, std::index_sequence_for<Args...>{});
  }
};

// TODO: Turn this into a generic wrapper for all other operations
template <class Fw, Fw fw, class... Args>
struct NestedTensorFunction_no_bw
    : public torch::autograd::Function<
          NestedTensorFunction_no_bw<Fw, fw, Args...>> {
  static typename c10::guts::infer_function_traits<Fw>::type::return_type
  forward(torch::autograd::AutogradContext* ctx, Args... args) {
    return fw(args...);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(false, "Backward not implemented.");
    return {};
  }
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

} // namespace at
