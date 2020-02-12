#pragma once
#include <nested_node.h>

namespace torch {
namespace nested_tensor {

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct _ListNestedTensor {
  _ListNestedTensor() = delete;
  _ListNestedTensor(TensorNode&& structure)
      : _structure(structure),
        _first_variable(
            get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                       : at::ones({})) {}
  int64_t element_size() {
    return _first_variable.element_size();
  }
  SizeNode nested_size() {
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
        _structure);
  }
  SizeNode nested_stride() {
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.strides()); },
        _structure);
  }
  _ListNestedTensor pin_memory() {
    return _ListNestedTensor(
        map([](at::Tensor tensor) { return tensor.pin_memory(); }, _structure));
  }
  _ListNestedTensor grad() {
    return _ListNestedTensor(
        map([](at::Tensor tensor) { return tensor.grad(); }, _structure));
  }
  _ListNestedTensor detach() {
    return _ListNestedTensor(
        map([](at::Tensor tensor) { return tensor.detach(); }, _structure));
  }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(map(
        [requires_grad](at::Tensor tensor) {
          return tensor.set_requires_grad(requires_grad);
        },
        _structure));
  }
  void backward(
      _ListNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    apply(
        [retain_graph, create_graph](at::Tensor tensor1, at::Tensor tensor2)
            -> void { tensor1.backward(tensor2, retain_graph, create_graph); },
        _structure,
        gradient.get_structure());
  }
  int64_t __len__() {
    return _structure.degree();
  }
  at::Tensor to_tensor() {
    return stack(flatten(_structure).vec());
  }
  int64_t nested_dim() {
    return _structure.height();
  }
  at::ScalarType scalar_type() {
    return _first_variable.scalar_type();
  }
  at::Backend backend() {
    return options().backend();
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
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + leaf.numel();
    };
    return reduce<decltype(fn), int64_t, at::Tensor>(_structure, fn, 0);
  }
  bool is_pinned() {
    return _first_variable.is_pinned();
  }
  bool is_contiguous() {
    return false;
  }
  TensorNode& get_structure() {
    return _structure;
  }
  const TensorNode& get_structure() const {
    return _structure;
  }
  // TODO: Implement these and call into them isntead of implementing them
  // separately in Variable dispatch functions.
  // _ListNestedTensor to - it's a pain due to the 100s of to overloads
  // separately in Variable dispatch functions.

 private:
  TensorNode _structure;
  at::Tensor _first_variable;
};
} // namespace nested_tensor
} // namespace torch
