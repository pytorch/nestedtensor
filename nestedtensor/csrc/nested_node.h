#pragma once
#include <torch/extension.h>
#include <Python.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;
// The implicit contract is that, if there are no children, variable_node is
// defined.
struct _NestedNode {
  _NestedNode() : _payload() {}
  _NestedNode(const std::vector<_NestedNode> children)
      : _children(children), _payload() {}
  _NestedNode(c10::IValue payload) : _payload(payload) {}
  inline bool is_leaf() const {
    return _children.size() == 0;
  }
  inline c10::IValue payload() const {
    return _payload;
  }
  inline const std::vector<_NestedNode> children() const {
    return _children;
  }
  inline _NestedNode children(size_t i) const {
    return _children[i];
  }
  inline const _NestedNode* children_data(size_t i) const {
    return _children.data() + i;
  }
  inline size_t degree() const {
    return _children.size();
  }

 private:
  const std::vector<_NestedNode> _children;
  // TODO: Make this const?
  // _VariableNode _variable_node;
  c10::IValue _payload;
};

inline PyObject* wrap_list(std::vector<PyObject*> list) {
  auto r = THPObjectPtr{PyTuple_New(list.size())};
  if (!r)
    throw python_error();
  for (size_t i = 0; i < list.size(); ++i) {
    PyTuple_SET_ITEM(r.get(), i, list[i]);
  }
  return r.release();
}

inline PyObject* wrap_nested_node(_NestedNode nested_node) {
  if (nested_node.is_leaf()) {
    return torch::jit::toPyObject(nested_node.payload()).release().ptr();
  } else {
    std::vector<PyObject*> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.push_back(wrap_nested_node(nested_node.children(i)));
    }
    return wrap_list(result);
  }
}

static std::string _NestedNode___str__(const _NestedNode& nested_node) {
  std::stringstream result;
  if (nested_node.is_leaf()) {
    PyObject* objectsRepresentation =
        PyObject_Str(THPVariable_Wrap(nested_node.payload().toTensor()));
    result << THPUtils_unpackString(objectsRepresentation);
    return result.str();
  } else {
    result << "nested_tensor([";
    result << std::endl;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result << "  ";
      result << _NestedNode___str__(nested_node.children(i));
      result << ",";
      result << std::endl;
    }
    result << "])";
    return result.str();
  }
}

static inline _NestedNode _get_structure(PyObject* tensors) {
  if (THPVariable_Check(tensors)) {
    auto variable = THPVariable_Unpack(tensors);
    return _NestedNode(variable);
  } else {
    std::vector<_NestedNode> meta_nodes;
    Py_ssize_t i, n;
    n = PyObject_Length(tensors);
    PyObject* item;
    if (n < 0) {
      throw python_error();
    }
    for (i = 0; i < n; i++) {
      item = PyList_GetItem(tensors, i);
      _NestedNode node = _get_structure(item);
      meta_nodes.push_back(node);
    }
    return _NestedNode(meta_nodes);
  }
}

static inline torch::autograd::Variable _get_first_tensor(PyObject* tensors) {
  if (THPVariable_Check(tensors)) {
    return THPVariable_Unpack(tensors);
  } else {
    return _get_first_tensor(PyList_GetItem(tensors, 0));
  }
}

static inline int64_t _numel(const _NestedNode& meta_node) {
  if (meta_node.is_leaf()) {
    return meta_node.payload().toTensor().numel();
  } else {
    int64_t result = 0;
    for (size_t i = 0; i < meta_node.degree(); i++) {
      result += _numel(meta_node.children(i));
    }
    return result;
  }
}

static inline at::Tensor _get_first_variable(_NestedNode nested_node) {
  const _NestedNode* start = &nested_node;
  while (!start->is_leaf()) {
    start = start->children_data(0);
  }
  if (!start->payload().isNone()) {
    return start->payload().toTensor();
  } else {
    return torch::ones({1});
  }
}

template <typename T, class F>
static inline T map(_NestedNode nested_node, F fn) {
  if (nested_node.is_leaf()) {
    // TODO: For now we assume the user doesn't want to apply her function if
    // the payload is None.
    T new_nested_node(fn(nested_node.payload()));
    return new_nested_node;
  } else {
    std::vector<T> new_children;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      new_children.push_back(T(map<T>(nested_node.children(i), fn)));
    }
    return T(new_children);
  }
}

template <class F>
static inline void apply2(
    _NestedNode nested_node1,
    _NestedNode nested_node2,
    F fn) {
  if (nested_node1.is_leaf()) {
    fn(nested_node1.payload().toTensor(), nested_node2.payload().toTensor());
  } else {
    for (size_t i = 0; i < nested_node1.degree(); i++) {
      apply2(nested_node1.children(i), nested_node2.children(i), fn);
    }
  }
}

static inline torch::autograd::Variable _NestedNode_to_tensor(
    const _NestedNode& nested_node) {
  if (nested_node.is_leaf()) {
    return nested_node.payload().toTensor();
  } else {
    std::vector<at::Tensor> variables;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      variables.push_back(_NestedNode_to_tensor(nested_node.children(i)));
    }
    return stack(variables);
  }
}

static inline bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const _NestedNode nested_node) {
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
    at::Tensor variable = nested_node.payload().toTensor();
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
