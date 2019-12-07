#pragma once

#include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/core/ivalue.h>
// #include <torch/csrc/Device.h>
// #include <torch/csrc/Dtype.h>
// #include <torch/csrc/Exceptions.h>
// #include <torch/csrc/autograd/python_variable.h>
// #include <torch/csrc/python_headers.h>
// #include <torch/csrc/tensor/python_tensor.h>
// #include <torch/csrc/utils/tensor_new.h>

// TODO:
// - HANDLE_TH_ERRORS
// - Python exception handling.
// - Implement NestedSize to avoid expensive python ops in *_nested_size/stride
// - map and apply functions that accepted JIT-ed functions to avoid unbind
// - don't make some functions static?
// - DEBUG enabled checking of constiuents

namespace torch {
namespace nested_tensor {

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
    // PyObject* fake_args = PyTuple_New(0);
    // PyObject* fake_kwargs = PyDict_New();
    // TODO: Update if python_variable updates it too
    // torch::tensor 
    return torch::ones({2, 2});
    // return torch::utils::legacy_tensor_ctor(
    //     torch::tensors::get_default_tensor_type_id(),
    //     torch::tensors::get_default_scalar_type(),
    //     fake_args,
    //     fake_kwargs);
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

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct TORCH_API _ListNestedTensor {
  _ListNestedTensor() = delete;
  _ListNestedTensor(_NestedNode structure)
      : _structure(structure),
        _first_variable(_get_first_variable(_structure)) {
    if (__len__() > 0) {
      TORCH_CHECK(
          _verify_variables(_first_variable, _structure),
          "Tensors don't line up.");
    }
  }
  int64_t element_size() {
    return _first_variable.element_size();
  }
  _NestedNode nested_size() {
    if (nested_dim() == 0) {
      return _NestedNode(at::IntArrayRef());
    }
    return map<_NestedNode>(
        _structure, [&](c10::IValue tensor) -> at::IntArrayRef {
          return tensor.toTensor().sizes();
        });
  }
  _NestedNode nested_stride() {
    if (nested_dim() == 0) {
      return _NestedNode(at::IntArrayRef());
    }
    return map<_NestedNode>(
        _structure, [&](c10::IValue tensor) -> at::IntArrayRef {
          return tensor.toTensor().strides();
        });
  }
  _ListNestedTensor to(
      at::TensorOptions options,
      bool non_blocking,
      bool copy,
      c10::optional<c10::MemoryFormat> memory_format) {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [&](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().to(
              options, non_blocking, copy, memory_format);
        }));
  }
  _ListNestedTensor to(
      at::ScalarType dtype,
      bool non_blocking,
      bool copy,
      c10::optional<c10::MemoryFormat> memory_format) {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [&](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().to(dtype, non_blocking, copy, memory_format);
        }));
  }
  _ListNestedTensor to(
      at::Device device,
      at::ScalarType dtype,
      bool non_blocking,
      bool copy,
      c10::optional<c10::MemoryFormat> memory_format) {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [&](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().to(
              device, dtype, non_blocking, copy, memory_format);
        }));
  }
  _ListNestedTensor pin_memory() {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().pin_memory();
        }));
  }
  _ListNestedTensor grad() {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().grad();
        }));
  }
  _ListNestedTensor detach() {
    return _ListNestedTensor(
        map<_NestedNode>(_structure, [](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().detach();
        }));
  }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(map<_NestedNode>(
        _structure, [requires_grad](c10::IValue tensor) -> at::Tensor {
          return tensor.toTensor().requires_grad_(requires_grad);
        }));
  }
  void backward(
      _ListNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    apply2(
        _structure,
        gradient.get_structure(),
        [retain_graph, create_graph](at::Tensor tensor1, at::Tensor tensor2) {
          tensor1.backward(tensor2, retain_graph, create_graph);
        });
  }
  int64_t __len__() {
    return _structure.degree();
  }
  std::string __str__();
  std::string __repr__();
  torch::autograd::Variable to_tensor() {
    return _NestedNode_to_tensor(_structure);
  }
  int64_t nested_dim() {
    const _NestedNode* start_structure = &_structure;
    int64_t depth = 0;
    while (!start_structure->is_leaf()) {
      depth++;
      start_structure = start_structure->children_data(0);
    }
    return depth;
  }
  at::ScalarType scalar_type() {
    return _first_variable.scalar_type();
  }
  at::Backend backend() {
    return _first_variable.type().backend();
  }
  at::Device device() {
    return _first_variable.device();
  }
  at::TensorOptions options() {
    return _first_variable.options();
  }
  bool requires_grad() {
    return _first_variable.requires_grad();
  }
  int64_t dim() {
    return _first_variable.dim() + nested_dim();
  }
  int64_t numel() {
    return _numel(_structure);
  }
  bool is_pinned() {
    return _first_variable.is_pinned();
  }
  bool is_contiguous() {
    return false;
  }
  _NestedNode get_structure() {
    return _structure;
  }
  // TODO: Implement these and call into them isntead of implementing them
  // separately in Variable dispatch functions.
  // _ListNestedTensor to - it's a pain due to the 100s of to overloads
  // py::tuple size(int64_t dim);
  // separately in Variable dispatch functions.
  // std::vector<py::object> unbind();
  // std::string __str__();
  // std::string __repr__();
  // py::tuple size(int64_t dim);

 private:
  const _NestedNode _structure;
  at::Tensor _first_variable;
};

void initialize_python_bindings();

} // namespace nested_tensor
} // namespace torch
