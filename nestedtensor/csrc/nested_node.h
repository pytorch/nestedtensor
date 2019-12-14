#pragma once
#include <Python.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>
#include <type_traits>

namespace torch {
namespace nested_tensor {

// TODO: Don't use using for namespaces in header!
using namespace torch::jit;
using namespace torch::autograd::utils;
template <typename T = c10::IValue>
struct NestedNode {
  NestedNode() : _is_leaf(true) {}
  NestedNode(const std::vector<NestedNode<T>> children)
      : _is_leaf(false), _children(children) {}
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

template <typename A>
inline py::object wrap_nested_node(NestedNode<A> nested_node) {
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

static std::string _NestedNode___str__(const TensorNode& nested_node) {
  std::stringstream result;
  result << "nested_tensor([";
  result << std::endl;
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      PyObject* objectsRepresentation =
          PyObject_Str(THPVariable_Wrap(nested_node.payload(i)));
      result << THPUtils_unpackString(objectsRepresentation);
    }
  } else {
    result << "  ";
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result << _NestedNode___str__(nested_node.children(i));
    }
    result << ",";
    result << std::endl;
  }
  result << "])";
  return result.str();
}

static IValue py_obj_to_ivalue(py::object py_obj) {
  auto inferred_type = tryToInferType(py_obj);
  if (!inferred_type.success()) {
    std::cerr << inferred_type.reason() << std::endl;
    throw python_error();
  }
  auto payload = toIValue(py_obj, inferred_type.type());
  return payload;
}

static inline TensorNode _get_tensor_structure(py::list py_obj) {
  // Empty list of Tensors
  if (py_obj.size() == 0) {
    return TensorNode();
  }
  IValue payload = py_obj_to_ivalue(py_obj);
  if (payload.isTensorList()) {
    // List of Tensors
    return TensorNode(payload.toTensorList());
  } else {
    // List of lists of Tensors
    std::vector<TensorNode> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      py::list py_obj_i = py::list(py_obj[i]);
      result.push_back(_get_tensor_structure(py_obj_i));
    }
    return TensorNode(result);
  }
}

static inline SizeNode _get_size_structure(py::list py_obj) {
  // Empty list of lists
  if (py_obj.size() == 0) {
    return SizeNode();
  }

  // List of empty lists
  py::list py_obj_0 = py_obj[0];
  if (py_obj_0.size() == 0) {
    c10::List<c10::List<int64_t>> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      result.push_back(c10::List<int64_t>());
    }
    return SizeNode(result);
  }

  // List of lists of numbers
  InferredType inferred_type = tryToInferType(py_obj[0]);
  if (inferred_type.success() && py_obj_to_ivalue(py_obj[0]).isIntList()) {
    c10::List<c10::List<int64_t>> result;
    for (size_t i = 0; i < py_obj.size(); i++) {
      result.push_back(py_obj_to_ivalue(py_obj[i]).toIntList());
    }
    return SizeNode(result);
  }

  // List of lists of lists...
  std::vector<SizeNode> result;
  for (size_t i = 0; i < py_obj.size(); i++) {
    py::list py_obj_i = py_obj[i];
    result.emplace_back(_get_size_structure(py_obj_i));
  }
  return SizeNode(result);
}

static inline int64_t nested_node_numel(
    const NestedNode<at::Tensor>& meta_node) {
  int64_t result = 0;
  if (meta_node.is_leaf()) {
    for (size_t i = 0; i < meta_node.size(); i++) {
      result += meta_node.payload(i).numel();
    }
  } else {
    for (size_t i = 0; i < meta_node.degree(); i++) {
      result += nested_node_numel(meta_node.children(i));
    }
  }
  return result;
}

static inline int64_t num_memory(
    c10::List<int64_t> size,
    c10::List<int64_t> stride) {
  if (size.size() == 0) {
    return 0;
  }
  return size[0] * stride[0];
}

static inline int64_t size_node_memory(
    SizeNode nested_size,
    SizeNode nested_stride) {
  int64_t result = 0;
  if (nested_size.is_leaf()) {
    for (size_t i = 0; i < nested_size.size(); i++) {
      result += num_memory(nested_size.payload(i), nested_stride.payload(i));
    }
  } else {
    for (size_t i = 0; i < nested_size.degree(); i++) {
      result +=
          size_node_memory(nested_size.children(i), nested_stride.children(i));
    }
  }
  return result;
}

template <typename A>
static NestedNode<A> get_first_leaf(NestedNode<A> nested_node) {
  const NestedNode<A>* start = &nested_node;
  while (!start->is_leaf()) {
    start = start->children_data(0);
  }
  return *start;
}

static inline at::Tensor _get_first_variable(TensorNode nested_node) {
  TensorNode leaf = get_first_leaf(nested_node);
  if (leaf.size()) {
    return leaf.payload(0);
  } else {
    return torch::ones({});
  }
}

template <typename B, typename A, typename F>
static inline NestedNode<A> map(NestedNode<B> nested_node, F fn) {
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
static inline void apply2(
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

static inline at::Tensor NestedNode_to_tensor(
    const NestedNode<at::Tensor>& nested_node) {
  std::vector<at::Tensor> variables;
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      variables.emplace_back(nested_node.payload(i));
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      variables.emplace_back(NestedNode_to_tensor(nested_node.children(i)));
    }
  }
  return stack(variables);
}

static inline bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const TensorNode nested_node) {
  // The attributes must match across all constiuents
  //
  // The NestedTensor's attributes then become that of its
  // constiuents.
  //
  // data must be a list of Tensors or NestedTensors
  //
  // Attributes:
  //     dim()
  //     layout
  //     device
  //     dtype
  //     requires_grad
  //     is_pinned()
  bool valid = true;
  if (nested_node.is_leaf()) {
    for (size_t i = 0; i < nested_node.size(); i++) {
      at::Tensor variable = nested_node.payload(i);
      // TODO: Add more checks?
      valid = valid && (variable.dim() == first_variable.dim());
      valid = valid && (variable.layout() == first_variable.layout());
      valid = valid && (variable.device() == first_variable.device());
      valid = valid && (variable.dtype() == first_variable.dtype());
      valid =
          valid && (variable.requires_grad() == first_variable.requires_grad());
      // NOTE: This is a very costly check! For now we'll let this to be enabled
      // manually. valid = valid && (variable_.is_pinned() ==
      // first_variable.is_pinned());
    }
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      valid =
          valid && _verify_variables(first_variable, nested_node.children(i));
    }
  }
  return valid;
}

} // namespace nested_tensor
} // namespace torch
