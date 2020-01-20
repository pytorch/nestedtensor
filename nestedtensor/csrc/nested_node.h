#pragma once
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace nested_tensor {

template <typename T = c10::IValue>
struct NestedNode {
  NestedNode() : _is_leaf(true) {}
  NestedNode(const std::vector<NestedNode<T>> children)
      : _is_leaf(false), _children(children) {}
  // NestedNode(const NestedNode&) = delete;
  NestedNode(c10::List<T> payload) : _is_leaf(true), _payload(payload) {}
  inline bool is_leaf() const {
    return _is_leaf;
  }
  inline T payload(size_t i) const {
    return _payload[i];
  }
  inline NestedNode<T> children(size_t i) const {
    return _children[i];
  }
  inline const NestedNode<T>* children_data(size_t i) const {
    return _children.data() + i;
  }
  inline size_t degree() const {
    return _children.size();
  }
  inline size_t size() const {
    return _payload.size();
  }

 private:
  bool _is_leaf;
  const std::vector<NestedNode<T>> _children;
  // TODO: Make this const?
  // _VariableNode _variable_node;
  c10::List<T> _payload;
};

template <typename T>
inline bool operator==(const NestedNode<T>& a, const NestedNode<T>& b) {
  if (a.is_leaf() != b.is_leaf()) {
    return false;
  }
  if (a.is_leaf()) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
      if (a.payload(i).size() != b.payload(i).size()) {
        return false;
      }
      for (size_t j = 0; j < a.payload(i).size(); j++) {
        if (a.payload(i)[j] != b.payload(i)[j]) {
          return false;
        }
      }
    }
  } else {
    if (a.degree() != b.degree()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
      if (a.children(i) != b.children(i)) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
inline bool operator!=(const NestedNode<T>& a, const NestedNode<T>& b) {
  return !(a == b);
}

using TensorNode = NestedNode<at::Tensor>;

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

using SizeNode = NestedNode<c10::List<int64_t>>;

std::string TensorNode___str__(
    const TensorNode& nested_node,
    const std::string& tabs = "");

std::string SizeNode___str__(
    const SizeNode& nested_node,
    const std::string name,
    const std::string& tabs = "");

c10::optional<c10::IValue> py_obj_to_ivalue(py::object py_obj);

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride);

int64_t size_node_memory(SizeNode nested_size, SizeNode nested_stride);

template <typename A>
py::object wrap_nested_node(NestedNode<A> nested_node) {
  std::vector<py::object> result;
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      result.push_back(torch::jit::toPyObject(nested_node.payload(i)));
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.push_back(wrap_nested_node(nested_node.children(i)));
    }
  }
  py::list result1 = py::cast(result);
  return result1;
}

at::Tensor NestedNode_to_tensor(const NestedNode<at::Tensor>& nested_node);

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node);

bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const TensorNode nested_node);

template <typename A>
inline c10::optional<A> get_first_leaf(NestedNode<A> nested_node) {
  if (nested_node.is_leaf() && nested_node.size() == 0) {
    return c10::nullopt;
  }
  const NestedNode<A>* start = &nested_node;
  while (!start->is_leaf()) {
    start = start->children_data(0);
  }
  return start->payload(0);
}

template <typename B, typename A, typename F>
inline NestedNode<A> map(NestedNode<B> nested_node, F fn) {
  if (nested_node.is_leaf()) {
    c10::List<A> result;
    for (size_t i = 0; i < nested_node.size(); i++) {
      result.emplace_back(fn(nested_node.payload(i)));
    }
    return NestedNode<A>(result);
  } else {
    std::vector<NestedNode<A>> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.emplace_back(map<B, A, F>(nested_node.children(i), fn));
    }
    return NestedNode<A>(result);
  }
}

// NOTE: Assuming all NestedNodes have same shape.
template <typename F, typename A, typename... B>
inline A reduce(NestedNode<B>... nested_node, F fn, A ident) {
  A result = ident;
  auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
  if (first_node.is_leaf()) {
    for (size_t i = 0; i < first_node.size(); i++) {
      result = fn(nested_node.payload(i)..., result);
    }
  } else {
    for (size_t i = 0; i < first_node.degree(); i++) {
      result = reduce<F, A, B...>(nested_node.children(i)..., fn, result);
    }
  }
  return result;
}

template <typename A, class F>
inline void apply(NestedNode<A> nested_node, F fn) {
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      fn(nested_node.payload(i));
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      apply(nested_node.children(i), fn);
    }
  }
}

template <typename A, class F>
inline void apply2(
    NestedNode<A> nested_node1,
    NestedNode<A> nested_node2,
    F fn) {
  if (nested_node1.is_leaf()) {
    for (size_t i = 0; i < nested_node1.size(); i++) {
      fn(nested_node1.payload(i), nested_node2.payload(i));
    }
  } else {
    for (size_t i = 0; i < nested_node1.degree(); i++) {
      apply2(nested_node1.children(i), nested_node2.children(i), fn);
    }
  }
}

template <typename T>
inline void aggregate_leafs(NestedNode<T> input, std::vector<T>& result) {
  if (input.is_leaf()) {
    for (size_t i = 0; i < input.size(); i++) {
      result.push_back(input.payload(i));
    }
  } else {
    for (size_t i = 0; i < input.degree(); i++) {
      aggregate_leafs<T>(input.children(i), result);
    }
  }
}

} // namespace nested_tensor
} // namespace torch
