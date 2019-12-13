#pragma once
#include <nested_node.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct _ListNestedTensor {
  _ListNestedTensor() = delete;
  _ListNestedTensor(TensorNode structure)
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
  SizeNode nested_size() {
    if (nested_dim() == 1 && _structure.degree() == 0) {
      return SizeNode();
    }
    return map<at::Tensor, c10::List<int64_t>>(
        _structure, [](at::Tensor tensor) -> c10::List<int64_t> {
          return c10::List<int64_t>(tensor.sizes());
        });
  }
  SizeNode nested_stride() {
    if (nested_dim() == 1 && _structure.degree() == 0) {
      return SizeNode();
    }
    return map<at::Tensor, c10::List<int64_t>>(
        _structure, [](at::Tensor tensor) -> c10::List<int64_t> {
          return c10::List<int64_t>(tensor.strides());
        });
  }
  _ListNestedTensor pin_memory() {
    return _ListNestedTensor(map<at::Tensor, at::Tensor>(
        _structure,
        [](at::Tensor tensor) -> at::Tensor { return tensor.pin_memory(); }));
  }
  _ListNestedTensor grad() {
    return _ListNestedTensor(map<at::Tensor, at::Tensor>(
        _structure,
        [](at::Tensor tensor) -> at::Tensor { return tensor.grad(); }));
  }
  _ListNestedTensor detach() {
    return _ListNestedTensor(map<at::Tensor, at::Tensor>(
        _structure,
        [](at::Tensor tensor) -> at::Tensor { return tensor.detach(); }));
  }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(map<at::Tensor, at::Tensor>(
        _structure, [requires_grad](at::Tensor tensor) -> at::Tensor {
          return tensor.set_requires_grad(requires_grad);
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
  at::Tensor to_tensor() {
    return NestedNode_to_tensor(_structure);
  }
  int64_t nested_dim() {
    const TensorNode* start_structure = &_structure;
    int64_t depth = 1;
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
    return nested_node_numel(_structure);
  }
  bool is_pinned() {
    return _first_variable.is_pinned();
  }
  bool is_contiguous() {
    return false;
  }
  TensorNode get_structure() {
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
  TensorNode _structure;
  at::Tensor _first_variable;
};
} // namespace nested_tensor
} // namespace torch
