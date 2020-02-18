#pragma once
#include <buffer_nested_tensor.h>
#include <list_nested_tensor.h>
// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>

namespace torch {
namespace nested_tensor {

template <typename T>
struct THPNestedNode {
  THPNestedNode(NestedNode<T> size_node, std::string name)
      : _size_node(size_node), _name(name) {}
  int64_t len() {
    return _size_node.degree();
  }
  std::string str() {
    return NestedNode___str__(
        _size_node, _name, [](c10::IValue payload, const std::string& tabs) {
          std::stringstream ss;
          ss << tabs << payload;
          return ss.str();
        });
  }
  const NestedNode<T>& get_node() const {
    return _size_node;
  }
  std::string get_name() {
    return _name;
  }

  py::object unbind() {
    std::vector<py::object> result;
    for (const auto& child : _size_node.unbind()) {
      if (child.height() == 0) {
        result.push_back(wrap_nested_node(child));
      } else {
        result.push_back(py::cast(THPNestedNode<T>(child, _name)));
      }
    }
    return py::cast(result);
  }

 private:
  NestedNode<T> _size_node;
  std::string _name;
};

using THPSizeNode = THPNestedNode<c10::List<int64_t>>;
using THPIntegerNode = THPNestedNode<int64_t>;
using THPTensorNode = THPNestedNode<at::Tensor>;
using THPIValueNode = THPNestedNode<c10::IValue>;

struct THPNestedTensor {
  THPNestedTensor() = delete;
  THPNestedTensor(_BufferNestedTensor data) : _data(data) {}
  THPNestedTensor(_ListNestedTensor data) : _data(data) {}
  at::Tensor get_buffer() {
    return _data.right().get_buffer();
  }
  int64_t element_size() {
    return _data.is_right() ? _data.right().element_size()
                            : _data.left().element_size();
  }
  pybind11::object getDtype();
  pybind11::object getLayout();
  pybind11::object getDevice();
  pybind11::object to_list() {
    auto node = _data.is_right() ? _data.right().get_structure()
                                 : _data.left().get_structure();
    return wrap_nested_node<at::Tensor, py::list>(node);
  }
  pybind11::object to_tuple() {
    auto node = _data.is_right() ? _data.right().get_structure()
                                 : _data.left().get_structure();
    return wrap_nested_node<at::Tensor, py::tuple>(node);
  }
  bool requires_grad() {
    return _data.is_right() ? _data.right().requires_grad()
                            : _data.left().requires_grad();
  }
  std::vector<c10::optional<int64_t>> size() {
    SizeNode tmp =
        map([](c10::IValue e) { return e.toIntList(); },
            this->nested_size().get_node());
    return construct_size(tmp);
  }
  // TODO: Not covered by 0.0.2 or 0.0.1!
  // NOTE: Returns a view
  // TODO: Advanced indexing
  // TODO: Tensor-wise select
  // TODO: Tuple support
  pybind11::object getitem(int64_t key) {
    py::object unbound_ = unbind();
    py::sequence unbound = py::cast<py::sequence>(unbound_);
    return unbound[key];
  }
  pybind11::object getitem(py::slice key) {
    py::object unbound_ = unbind();
    py::sequence unbound = py::cast<py::sequence>(unbound_);
    return unbound[key];
  }
  pybind11::object unbind();
  THPIValueNode nested_size();
  THPIValueNode nested_stride();
  THPIValueNode nested_size(c10::optional<int64_t> index);
  THPIValueNode nested_stride(c10::optional<int64_t> index);
  THPNestedTensor requires_grad_(pybind11::bool_ requires_grad_) {
    bool requires_grad = requires_grad_;
    if (_data.is_right()) {
      return THPNestedTensor(_data.right().requires_grad_(requires_grad));
    }
    return THPNestedTensor(_data.left().requires_grad_(requires_grad));
  }
  THPNestedTensor grad() {
    if (_data.is_right()) {
      return THPNestedTensor(_data.right().grad());
    }
    return THPNestedTensor(_data.left().grad());
  }
  THPNestedTensor detach() {
    if (_data.is_right()) {
      return THPNestedTensor(_data.right().detach());
    }
    return THPNestedTensor(_data.left().detach());
  }
  THPNestedTensor pin_memory() {
    if (_data.is_right()) {
      return THPNestedTensor(_data.right().pin_memory());
    }
    return THPNestedTensor(_data.left().pin_memory());
  }
  std::string str();
  int64_t len() {
    return _data.is_right() ? _data.right().__len__() : _data.left().__len__();
  }
  bool is_pinned() {
    return _data.is_right() ? _data.right().is_pinned()
                            : _data.left().is_pinned();
  }
  int64_t nested_dim() {
    if (_data.is_right()) {
      return _data.right().nested_dim();
    }
    return _data.left().nested_dim();
  }
  int64_t dim() {
    return _data.is_right() ? _data.right().dim() : _data.left().dim();
  }
  int64_t numel() {
    return _data.is_right() ? _data.right().numel() : _data.left().numel();
  }
  at::Tensor to_tensor() {
    return _data.is_right() ? _data.right().to_tensor()
                            : _data.left().to_tensor();
  }
  THPNestedTensor contiguous();
  bool is_contiguous() const {
    return _data.is_right() ? _data.right().is_contiguous()
                            : _data.left().is_contiguous();
  }

  c10::either<_ListNestedTensor, _BufferNestedTensor> _data;
};

} // namespace nested_tensor
} // namespace torch
