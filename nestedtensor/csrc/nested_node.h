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

// TODO: Need to fix indentation.
std::string _NestedNode___str__(const TensorNode& nested_node);
c10::optional<c10::IValue> py_obj_to_ivalue(py::object py_obj);

int64_t nested_node_numel(const TensorNode& meta_node);

bool all_contiguous(const TensorNode& meta_node);

bool all_size_equal(const SizeNode& nested_size);

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride);

int64_t size_node_memory(SizeNode nested_size, SizeNode nested_stride);

at::Tensor _get_first_variable(TensorNode nested_node);

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

bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const TensorNode nested_node);

template <typename A>
inline NestedNode<A> get_first_leaf(NestedNode<A> nested_node) {
  const NestedNode<A>* start = &nested_node;
  while (!start->is_leaf()) {
    start = start->children_data(0);
  }
  return *start;
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
inline void apply2(NestedNode<A> nested_node1, NestedNode<A> nested_node2, F fn) {
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

} // namespace nested_tensor
} // namespace torch
