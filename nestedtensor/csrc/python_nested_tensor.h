#pragma once
#include <nested_tensor.h>
#include <py_utils.h>
#include <utils/python_nested_node.h>
// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>

namespace torch {
namespace nested_tensor {

using THPSizeNode = THPNestedNode<c10::List<int64_t>>;
using THPIntegerNode = THPNestedNode<int64_t>;
using THPTensorNode = THPNestedNode<at::Tensor>;
using THPIValueNode = THPNestedNode<c10::IValue>;

struct THPNestedTensor {
  THPNestedTensor() = delete;
  THPNestedTensor(NestedTensor data) : _data(data) {}
  at::Tensor get_buffer() {
    return (*_data.get_buffer());
  }
  int64_t element_size() {
    return _data.element_size();
  }
  pybind11::object getDtype();
  pybind11::object getLayout();
  pybind11::object getDevice();
  pybind11::object to_list() {
    auto node = _data.get_structure();
    return wrap_nested_node<at::Tensor, py::list>(node);
  }
  pybind11::object to_tuple() {
    auto node = _data.get_structure();
    return wrap_nested_node<at::Tensor, py::tuple>(node);
  }
  bool requires_grad() {
    return _data.requires_grad();
  }
  std::vector<c10::optional<int64_t>> size() {
    return _data.size();
  }
  // TODO: Not covered by 0.0.2 or 0.0.1!
  // NOTE: Returns a view
  // TODO: Advanced indexing
  // TODO: Tensor-wise select
  // TODO: Tuple support
  pybind11::object getitem(int64_t key) {
    return unbind(0)[key];
  }
  pybind11::object getitem(py::slice key) {
    py::list unbound = py::cast(unbind(0));
    return unbound[key];
  }
  std::vector<pybind11::object> unbind(int64_t dim);
  THPIValueNode nested_size();
  THPIValueNode nested_stride();
  THPIValueNode nested_size(c10::optional<int64_t> index);
  THPIValueNode nested_stride(c10::optional<int64_t> index);
  THPNestedTensor requires_grad_(pybind11::bool_ requires_grad_) {
    bool requires_grad = requires_grad_;
    return THPNestedTensor(_data.requires_grad_(requires_grad));
  }
  THPNestedTensor grad() {
    return THPNestedTensor(_data.grad());
  }
  THPNestedTensor detach() {
    return THPNestedTensor(_data.detach());
  }
  THPNestedTensor pin_memory() {
    return THPNestedTensor(_data.pin_memory());
  }
  std::string str();
  int64_t len() {
    return _data.__len__();
  }
  bool is_pinned() {
    return _data.is_pinned();
  }
  int64_t nested_dim() {
    return _data.nested_dim();
  }
  int64_t dim() {
    return _data.dim();
  }
  int64_t numel() {
    return _data.numel();
  }
  py::object to_tensor(c10::optional<int64_t>);
  THPNestedTensor contiguous() {
    return _data.contiguous();
  }
  bool is_contiguous() const {
    return _data.is_contiguous();
  }
  NestedTensor& data() {
    return _data;
  }
  const NestedTensor& data() const {
    return _data;
  }

 private:
  NestedTensor _data;
};

} // namespace nested_tensor
} // namespace torch
