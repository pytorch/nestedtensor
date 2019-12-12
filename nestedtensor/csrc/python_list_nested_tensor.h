#pragma once
#include <Python.h>
#include <list_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/extension.h>

// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>
// TODO: Make sure constructor requires list!

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;
namespace py = pybind11;

struct THP_ListNestedTensor {
  THP_ListNestedTensor() = delete;
  THP_ListNestedTensor(py::object list)
      : _data(_ListNestedTensor(_get_tensor_structure(list))) {}
  THP_ListNestedTensor(_ListNestedTensor data) : _data(data) {}
  int64_t element_size() {
    return _data.element_size();
  }
  py::object nested_size() {
    return 
        wrap_nested_node(_data.nested_size());
  }
  py::object nested_stride() {
    return 
        wrap_nested_node(_data.nested_stride());
  }
  THP_ListNestedTensor pin_memory() {
    return THP_ListNestedTensor(_data.pin_memory());
  }
  THP_ListNestedTensor grad() {
    return THP_ListNestedTensor(_data.grad());
  }
  THP_ListNestedTensor detach() {
    return THP_ListNestedTensor(_data.detach());
  }
  THP_ListNestedTensor requires_grad_(py::bool_ requires_grad) {
    return THP_ListNestedTensor(_data.requires_grad_(requires_grad));
  }
  int64_t nested_dim() {
    return _data.nested_dim();
  }
  int64_t dim() {
    return _data.dim();
  }
  bool is_contiguous() {
    return _data.is_contiguous();
  }
  bool is_pinned() {
    return _data.is_pinned();
  }
  bool requires_grad() {
    return _data.requires_grad();
  }
  int64_t numel() {
    return _data.numel();
  }
  int64_t len() {
    return _data.__len__();
  }
  at::Tensor to_tensor() {
    return _data.to_tensor();
  }
  // NOTE: Don't delete this. repr is an important concept, this
  // implementation is just faulty due to torch.Tensor.__repr__
  // TODO: Assuming that there is no difference in __str__ and __repr__ for
  // torch.Tensor.
  std::string str() {
    return _NestedNode___str__(_data.get_structure());
  }
  py::object getDtype() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getDtype(_data.scalar_type())));
  }
  py::object getLayout() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getLayout(_data.backend())));
  }
  py::object getDevice() {
    return toPyObject(_data.device());
  }
  _ListNestedTensor data() {
    return _data;
  }
  void backward(
      THP_ListNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    _data.backward(gradient.data(), retain_graph, create_graph);
  }

 private:
  _ListNestedTensor _data;
};
} // namespace nested_tensor
} // namespace torch
