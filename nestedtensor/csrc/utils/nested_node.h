#pragma once
#include <ATen/core/List.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/Optional.h>
#include <c10/util/TypeList.h>

namespace torch {
namespace nested_tensor {

// NOTE: For comparisons please use the map and reduce
// functions to define what you mean by equal, etc. on your own
// There can be ambiguity in the depth of comparison and
// even in the value (should it construct a new tree or
// return a single value).
template <typename T>
struct NestedNode {
  // NestedNode() : _is_leaf(false), _height(1) {}
  NestedNode() = delete;
  explicit NestedNode(std::vector<NestedNode<T>>&& children)
      : _is_leaf(false), _children(children), _height(1) {
    for (const auto& child : children) {
      if (child.height() + 1 > _height) {
        _height = child.height() + 1;
      }
    }
    // for (const auto& child : children) {
    //   TORCH_CHECK(
    //       child.height() == _height - 1,
    //       "internal error: expected a full tree.");
    // }
  }
  // NestedNode(NestedNode&) = delete;
  // NestedNode(const NestedNode&) = delete;
  // NestedNode& operator=(NestedNode) = delete;
  explicit NestedNode(T&& payload) : _is_leaf(true), _payload(payload), _height(0) {}
  inline bool is_leaf() const {
    return _is_leaf;
  }
  inline size_t degree() const {
    return _children.size();
  }
  inline int64_t height() const {
    return _height;
  }
  inline const std::vector<NestedNode<T>> unbind() const {
    return _children;
  }
  inline NestedNode<T> children(size_t i) const {
    return _children[i];
  }
  inline const T& payload() const {
    return _payload;
  }
  inline T& payload() {
    return _payload;
  }

 private:
  bool _is_leaf;
  std::vector<NestedNode<T>> _children;
  // TODO: Make this const?
  // _VariableNode _variable_node;
  T _payload;
  int64_t _height;
};

// TODO: Should have specialized construction check that all payloads are of
// same size for SizeNode
using SizeNode = NestedNode<std::vector<int64_t>>;
using IntegerNode = NestedNode<int64_t>;
using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;

template <typename A>
inline c10::optional<A> get_first_leaf(NestedNode<A> nested_node) {
  if (nested_node.is_leaf()) {
    return nested_node.payload();
  }
  if (nested_node.degree() == 0) {
    return c10::nullopt;
  }
  for (const auto& child : nested_node.unbind()) {
    if (auto result = get_first_leaf(child)) {
      return result;
    }
  }
  return c10::nullopt;
}

template <class F, class A, class TypeList>
class _map;

template <class F, class A, class... Args>
class _map<F, A, c10::guts::typelist::typelist<Args...>> {
 public:
  // NOTE: We must move F to avoid copying objects if it is a lambda with
  // captures.
  static NestedNode<A> function(
      F&& fn,
      const NestedNode<Args>&... nested_node) {
    size_t degree = 0;
    bool all_leaf = true;
    c10::guts::tuple_map(
        std::forward_as_tuple(nested_node...), [&all_leaf, &degree](auto n) {
          all_leaf = all_leaf && (n.is_leaf());
          if (degree > 1 && n.degree() > 1) {
            TORCH_CHECK(degree == n.degree(), "NestedNodes don't broadcast.");
          }
          if (n.degree() > degree) {
            degree = n.degree();
          }
          return nullptr;
        });
    if (all_leaf) {
      return NestedNode<A>(std::forward<F>(fn)(nested_node.payload()...));
    }
    std::vector<NestedNode<A>> result;
    for (size_t i = 0; i < degree; i++) {
      std::tuple<NestedNode<Args>...> children = c10::guts::tuple_map(
          std::forward_as_tuple(nested_node...), [&i](auto a) {
            static_assert(
                c10::guts::is_instantiation_of<NestedNode, decltype(a)>::value,
                "Internal error.");
            if (a.is_leaf()) {
              return a;
            }
            if (a.degree() == 1 && a.height() > 0) {
              return a.children(0);
            }
            TORCH_CHECK(a.degree() > 0, "Internal assert.");
            return a.children(i);
          });
      // TODO: Due to the experiences with to_vector and the inversion I'm a bit
      // wary of apply but I haven't been able to reproduce the  argument
      // inversion behavior in other contexts.
      c10::guts::apply(
          [&result, &fn](NestedNode<Args>... filtered) {
            result.emplace_back(function(std::forward<F>(fn), filtered...));
          },
          std::move(children));
    }
    return NestedNode<A>(std::move(result));
  }
};

// NOTE: Assuming all NestedNodes have same shape.
// TODO: Add check
// TODO: Add static assert to verify lambda arguments match nested_node types
// TODO: Do we want broadcasting?
template <class F, class... B>
static inline NestedNode<
    typename c10::guts::infer_function_traits<F>::type::return_type>
map(F&& fn, const NestedNode<B>&... nested_node) {
  return _map<
      F,
      typename c10::guts::infer_function_traits<F>::type::return_type,
      typename c10::guts::infer_function_traits<F>::type::parameter_types>::
      function(std::forward<F>(fn), nested_node...);
}

template <typename A>
inline std::vector<A> flatten(NestedNode<A> nested_node) {
  if (nested_node.is_leaf()) {
    std::vector<A> result;
    result.push_back(nested_node.payload());
    return result;
  } else {
    std::vector<A> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      for (auto tmp : flatten<A>(nested_node.children(i))) {
        result.push_back(std::move(tmp));
      }
    }
    return result;
  }
}

template <class R, class A>
inline std::pair<int64_t, NestedNode<R>> _unflatten(
    const NestedNode<A>& structure,
    const std::vector<R>& content,
    int64_t index) {
  if (structure.is_leaf()) {
    R tmp = content[index];
    return std::pair<int64_t, NestedNode<R>>(
        index + 1, NestedNode<R>(std::move(tmp)));

  } else {
    std::vector<NestedNode<R>> result;
    for (size_t i = 0; i < structure.degree(); i++) {
      auto result_i = _unflatten<R, A>(structure.children(i), content, index);
      index = std::get<0>(result_i);
      result.emplace_back(std::get<1>(result_i));
    }
    return std::pair<int64_t, NestedNode<R>>(
        index, NestedNode<R>(std::move(result)));
  }
}

// NOTE: structure is only used as a shape guidance and its content doesn't
// matter. This function uses structure and content to create a new NestedNode
// with the same shape as structure and content distributed in-order
template <class R, class A>
inline NestedNode<R> unflatten(
    NestedNode<A> structure,
    std::vector<R> content) {
  auto _result = _unflatten<R, A>(structure, content, 0);
  return std::get<1>(_result);
}

template <class A>
inline std::vector<NestedNode<A>> unzip(
    const NestedNode<std::vector<A>>& structure) {
  if (structure.is_leaf()) {
    std::vector<NestedNode<A>> results;
    std::vector<A> payload = structure.payload();
    for (size_t i = 0; i < payload.size(); i++) {
      results.push_back(NestedNode<A>(std::move(payload[i])));
    }
    return results;
  } else {
    std::vector<std::vector<NestedNode<A>>> result;
    for (size_t i = 0; i < structure.degree(); i++) {
      std::vector<NestedNode<A>> unzipped = unzip(structure.children(i));
      for (size_t j = 0; j < unzipped.size(); j++) {
        if (j >= result.size()) {
          result.resize(j + 1);
        }
        result[j].push_back(unzipped[j]);
      }
    }
    std::vector<NestedNode<A>> wrapped_result;
    for (size_t i = 0; i < result.size(); i++) {
      wrapped_result.push_back(NestedNode<A>(std::move(result[i])));
    }
    return wrapped_result;
  }
}

template <class A>
inline NestedNode<std::vector<A>> zip(
    const std::vector<NestedNode<A>>& structures) {
  bool all_leaf = true;
  for (const auto& node : structures) {
    all_leaf = all_leaf && node.is_leaf();
  }
  if (all_leaf) {
    std::vector<A> results;
    for (size_t i = 0; i < structures.size(); i++) {
      results.push_back(structures[i].payload());
    }
    return NestedNode<std::vector<A>>(std::move(results));
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (const auto& node : structures) {
      if (!node.is_leaf()) {
        if (num_children > 0) {
          broadcastable = broadcastable && (num_children == node.degree());
        } else {
          num_children = node.degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<NestedNode<std::vector<A>>> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<NestedNode<A>> tmp;
      for (const auto& node : structures) {
        if (node.is_leaf()) {
          tmp.push_back(node);
        } else {
          tmp.push_back(node.children(i));
        }
      }
      result.push_back(zip(tmp));
    }
    return NestedNode<std::vector<A>>(std::move(result));
  }
}

// TODO: Assuming all NestedNodes have same shape.
template <typename F, typename A, typename... B>
inline typename c10::guts::infer_function_traits<F>::type::return_type reduce(
    F fn,
    A ident,
    NestedNode<B>... nested_node) {
  auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
  if (first_node.is_leaf()) {
    return fn(nested_node.payload()..., ident);
  }
  A result = ident;
  for (size_t i = 0; i < first_node.degree(); i++) {
    result = reduce<F, A, B...>(fn, result, nested_node.children(i)...);
  }
  return result;
}

template <class F, class TypeList>
class _apply;

template <class F, class... Args>
class _apply<F, c10::guts::typelist::typelist<Args...>> {
 public:
  // NOTE: We must move F to avoid copying objects if it is a lambda with
  // captures.
  static void function(F&& fn, NestedNode<Args>... nested_node) {
    size_t degree = 0;
    bool all_leaf = true;
    c10::guts::tuple_map(
        std::forward_as_tuple(nested_node...), [&all_leaf, &degree](auto n) {
          all_leaf = all_leaf && (n.is_leaf());
          if (degree == 0 && n.degree() > 0) {
            degree = n.degree();
          }
          if (degree > 0 && n.degree() > 0) {
            TORCH_CHECK(degree == n.degree(), "NestedNodes don't broadcast.");
          }
          return nullptr;
        });
    if (all_leaf) {
      std::forward<F>(fn)(nested_node.payload()...);
    } else {
      for (size_t i = 0; i < degree; i++) {
        std::tuple<NestedNode<Args>...> children = c10::guts::tuple_map(
            std::forward_as_tuple(nested_node...), [&i](auto a) {
              static_assert(
                  c10::guts::is_instantiation_of<NestedNode, decltype(a)>::
                      value,
                  "Internal error.");
              if (a.is_leaf()) {
                return a;
              }
              if (a.degree() == 1) {
                return a.children(0);
              }
              TORCH_CHECK(a.degree() > 0, "Internal assert.");
              return a.children(i);
            });
        c10::guts::apply(
            [&fn](NestedNode<Args>... filtered) {
              function(std::forward<F>(fn), filtered...);
            },
            std::move(children));
      }
    }
  }
};

// NOTE: Assuming all NestedNodes have same shape.
// TODO: Add check that all shapes match
// TODO: Add static assert to verify lambda arguments match nested_node types
// TODO: Do we want broadcasting?
// TODO: Add check that lambda returns void
template <class F, class... A>
static inline void apply(F&& fn, NestedNode<A>... nested_node) {
  _apply<
      F,
      c10::guts::typelist::map_t<
          std::remove_reference_t,
          typename c10::guts::infer_function_traits<F>::type::
              parameter_types>>::function(std::forward<F>(fn), nested_node...);
}

namespace impl {

inline std::vector<int64_t> _cont_stride(std::vector<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return std::vector<int64_t>(stride);
}

inline std::vector<int64_t> _cont_stride(int64_t* size_ptr, int64_t size) {
  std::vector<int64_t> size_vector(size_ptr, size_ptr + size);
  return _cont_stride(size_vector);
}

inline bool _is_cont_stride(int64_t* size, int64_t* stride, size_t length) {
  int64_t p = 1;
  size_t p_i = length;
  for (size_t i = 0; i < length; i++) {
    p_i--;
    if (p != stride[p_i]) {
      return false;
    }
    p *= size[p_i];
  }
  return true;
}

inline int64_t num_memory(
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  int64_t result = 1;
  for (size_t i = 0; i < size.size(); i++) {
    result = result + ((size[i] - 1) * stride[i]);
  }
  return result;
}

inline int64_t num_memory(
    int64_t* size_ptr,
    int64_t* stride_ptr,
    int64_t size) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  int64_t result = 1;
  for (size_t i = 0; i < size; i++) {
    result = result + ((size_ptr[i] - 1) * stride_ptr[i]);
  }
  return result;
}
} // namespace impl

// Remove singleton nodes across given level.
template <class A>
inline NestedNode<A> squeeze(
    NestedNode<A> structure,
    int64_t level,
    bool keep_dim) {
  if (level <= 0) {
    if (keep_dim) {
      std::vector<NestedNode<A>> children;
      children.push_back(structure.children(0));
      return NestedNode<A>(std::move(children));
    }
    return structure.children(0);
  }
  return NestedNode<A>(squeeze(structure, level - 1, keep_dim));
}

} // namespace nested_tensor
} // namespace torch
