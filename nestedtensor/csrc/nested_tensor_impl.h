#pragma once
#include <ATen/ATen.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <c10/util/Metaprogramming.h>
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

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutograd;

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

template <class F, class... A>
static inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  return wrap_tensor_node(
      map(std::move(fn), get_nested_tensor_structure(a)...));
}

// The approach here is quite "simple". There are six different stages to this.
// 1. We take the input NestedTensor whose constituents are, by design, required to not track gradients.
// Only the NestedTensor as a whole is allowed to track that information.
// 2. We take that NestedTensor and create a copy, i.e. a new NestedTensor, where the gradients do track
// gradients. This is not a valid NestedTensor outside the context of this function and in the future we 
// might decide to pick a different container, maybe even a flat list, for this purpose.
// 3. We set these constiuents of the new NestedTensor to track gradients. A very important point here is
// that within a custom autograd Function AutoGradMode is *disabled*, because we're defining a new elementary
// operation within the Autograd graph and aren't appending to it. We're effectively creating a subgraph
// for the purpose of this operation here that isn't connect to the overall graph that corresponds to
// NestedTensor operations.
// 4. We apply the differentiable function that was passed as an argument to each constiuents of the NestedTensor
// from step 3 again while enabling AutoGradMode. We will again get a NestedTensor where the constituents track
// gradients. To make sure we actually return a valid NestedTensor we detach this information for our return
// value and save the NestedTensor from this step only for the backward pass.
// 5. This step does the actual detach of the constituents
// 6. This step then returns the NestedTensor from step 5.
template <typename F, class... A>
struct NestedTensorFunction_mapper
    : public torch::autograd::Function<NestedTensorFunction_mapper<F, A...>> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      F&& fn,
      // 1. Original NestedTensors
      A... input) {
    std::cout << "Calling: " << typeid(F).name() << std::endl;
    auto autograd_input_tuple_ =
        c10::guts::tuple_map(std::tuple<A...>(input...), [](at::Tensor t) {
          apply_nested_tensor(
              [](at::Tensor& ti) {
                TORCH_CHECK(
                    !ti.requires_grad(),
                    "Input constituents shouldn't require gradients.");
              },
              t);
          if (t.requires_grad()) {
            return map_nested_tensor(
                // 2. Constituents of NestedTensors
                [](at::Tensor ti) {
                  AutoGradMode autogradmode(true);
                  // TODO: Don't apply this if the corresponding NestedTensor
                  // doesn't require a gradient.
                  ti.alias().requires_grad_();
                  // 3. Alias to constituents that do requires gradients
                  return ti;
                },
                t);
          }
          return t;
        });
    auto autograd_input_tuple = autograd_input_tuple_;

    auto tmp1 =
        c10::guts::tuple_map(std::tuple<A...>(input...), [](at::Tensor t) {
          apply_nested_tensor(
              [](at::Tensor& ti) {
                TORCH_CHECK(
                    !ti.requires_grad(),
                    "Input constituents shouldn't require gradients.");
              },
              t);
          return t;
        });

    // 4. Output of differentiable function given Tensor from step 3.
    at::Tensor autograd_output = c10::guts::apply(
        [&fn](A... a) {
          return map_nested_tensor(
              [&](A... t) {
                AutoGradMode autogradmode(true);
                return fn(t...);
              },
              a...);
        },
        std::move(autograd_input_tuple_));

    auto tmp2 =
        c10::guts::tuple_map(std::tuple<A...>(input...), [](at::Tensor t) {
          apply_nested_tensor(
              [](at::Tensor& ti) {
                TORCH_CHECK(
                    !ti.requires_grad(),
                    "Input constituents shouldn't require gradients.");
              },
              t);
          return t;
        });

    ctx->saved_data["0"] = autograd_input_tuple;
    ctx->saved_data["1"] = autograd_output;

    // 5. Constituents of output NestedTensor
    auto output = map_nested_tensor(
        [](at::Tensor t) { return t.detach(); }, autograd_output);

    auto tmp3 =
        c10::guts::tuple_map(std::tuple<A...>(input...), [](at::Tensor t) {
          apply_nested_tensor(
              [](at::Tensor& ti) {
                TORCH_CHECK(
                    !ti.requires_grad(),
                    "Input constituents shouldn't require gradients.");
              },
              t);
          return t;
        });


    // 6. Output NestedTensor
    return output;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output_) {
    std::cout << "grad_output_.size(): " << grad_output_.size() << std::endl;
    auto autograd_input_tuple = ctx->saved_data["0"].toTuple();
    auto autograd_output = ctx->saved_data["1"].toTensor();
    auto grad_input = map_nested_tensor(
        [](at::Tensor r, at::Tensor i, at::Tensor g) {
          // TODO: Might have to retain graph in many to one settings.
          return torch::autograd::grad({r}, {i}, {g})[0];
        },
        autograd_output,
        //        autograd_input_tuple[0].toTensor(),
        autograd_input_tuple->elements()[0].toTensor(),
        grad_output_[0]);

    at::Tensor undef;
    // at::Tensor grad_input;
    // grad_input.insert(grad_input.begin(), undef);
    // return grad_input;
    return {undef, grad_input};
  }
};

template <class F, class... A>
static inline at::Tensor autograd_map_nested_tensor(F&& fn, A... a) {
  return NestedTensorFunction_mapper<F, A...>::apply(std::move(fn), a...);
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
  Tensor requires_grad_(bool requires_grad) {
    apply(
        [requires_grad](at::Tensor& tensor) -> void {
          tensor.set_requires_grad(requires_grad);
        },
        get_structure());
    return at::detail::make_tensor<NestedTensorImpl>(_structure);
  }
  bool requires_grad() const {
    return _first_variable.requires_grad();
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

} // namespace at
