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
    if (_size_node.is_leaf()) {
      return _size_node.size();
    } else {
      return _size_node.degree();
    }
  }
  std::string str() {
    return NestedNode___str__(
        _size_node, _name, [](c10::IValue payload, const std::string& tabs) {
          std::stringstream ss;
          ss << "\n" << tabs << payload;
          return ss.str();
        });
  }
  const NestedNode<T>& get_node() {
    return _size_node;
  }
  std::string get_name() {
    return _name;
  }

  std::vector<py::object> unbind() {
    std::vector<py::object> result;
    if (_size_node.is_leaf()) {
      for (size_t i = 0; i < _size_node.size(); i++) {
        result.push_back(torch::jit::toPyObject(_size_node.payload(i)));
      }
    } else {
      for (size_t i = 0; i < _size_node.degree(); i++) {
        result.push_back(
            py::cast(THPNestedNode<T>(_size_node.children(i), _name)));
      }
    }
    return result;
  }

 private:
  NestedNode<T> _size_node;
  std::string _name;
};

using THPSizeNode = THPNestedNode<c10::List<int64_t>>;
using THPIntegerNode = THPNestedNode<int64_t>;
using THPTensorNode = THPNestedNode<at::Tensor>;

template <class Result, class F>
static inline Result data_map(
    c10::either<_ListNestedTensor, _BufferNestedTensor>& data,
    F&& fn) {
  return data.map<Result>(std::forward<F>(fn), std::forward<F>(fn));
}

struct THPNestedTensor {
  THPNestedTensor() = delete;
  THPNestedTensor(_BufferNestedTensor data) : _data(data) {}
  THPNestedTensor(_ListNestedTensor data) : _data(data) {}
  at::Tensor get_buffer() {
    return _data.right().get_buffer();
  }
  int64_t element_size() {
    return data_map<int64_t>(
        _data, [](auto data) { return data.element_size(); });
  }
  pybind11::object getDtype();
  pybind11::object getLayout();
  pybind11::object getDevice();
  bool requires_grad() {
    return data_map<bool>(
        _data, [](auto data) { return data.requires_grad(); });
  }
  c10::either<_ListNestedTensor, _BufferNestedTensor> data() {
    return _data;
  }
  std::vector<c10::optional<int64_t>> size() {
    return construct_size(this->nested_size().get_node());
  }
  THPSizeNode nested_size() {
    return THPSizeNode(
        data_map<SizeNode>(_data, [](auto data) { return data.nested_size(); }),
        "NestedSize");
  }
  THPSizeNode nested_stride() {
    return THPSizeNode(
        data_map<SizeNode>(
            _data, [](auto data) { return data.nested_stride(); }),
        "NestedStride");
  }
  THPNestedTensor requires_grad_(pybind11::bool_ requires_grad_) {
    bool requires_grad = requires_grad_;
    return THPNestedTensor(
        data_map<THPNestedTensor>(_data, [requires_grad](auto data) {
          return data.requires_grad_(requires_grad);
        }));
  }
  THPNestedTensor grad() {
    return data_map<THPNestedTensor>(
        _data, [](auto data) { return THPNestedTensor(data.grad()); });
  }
  THPNestedTensor detach() {
    return data_map<THPNestedTensor>(
        _data, [](auto data) { return THPNestedTensor(data.detach()); });
  }
  THPNestedTensor pin_memory() {
    return data_map<THPNestedTensor>(
        _data, [](auto data) { return THPNestedTensor(data.pin_memory()); });
  }
  std::string str() {
    return data_map<std::string>(_data, [](auto data) {
      return NestedNode___str__(
          data.get_structure(),
          "nested_tensor",
          [](c10::IValue payload, const std::string& tabs) {
            std::vector<std::string> tokens = split_str(
                THPUtils_unpackString(
                    PyObject_Str(THPVariable_Wrap(payload.toTensor()))),
                "\n");
            std::string result;
            for (const std::string& token : tokens) {
              result = result + "\n" + tabs + token;
            }
            return result;
          });
    });
  }
  int64_t len() {
    return data_map<int64_t>(_data, [](auto data) { return data.__len__(); });
  }
  bool is_pinned() {
    return data_map<bool>(_data, [](auto data) { return data.is_pinned(); });
  }
  int64_t nested_dim() {
    return data_map<int64_t>(
        _data, [](auto data) { return data.nested_dim(); });
  }
  int64_t dim() {
    return data_map<int64_t>(_data, [](auto data) { return data.dim(); });
  }
  int64_t numel() {
    return data_map<int64_t>(_data, [](auto data) { return data.numel(); });
  }
  at::Tensor to_tensor() {
    return data_map<at::Tensor>(
        _data, [](auto data) { return data.to_tensor(); });
  }
  THPNestedTensor contiguous();
  bool is_contiguous() {
    return data_map<bool>(
        _data, [](auto data) { return data.is_contiguous(); });
  }
  TensorNode get_structure() {
    return data_map<TensorNode>(
        _data, [](auto data) { return data.get_structure(); });
  }

 private:
  c10::either<_ListNestedTensor, _BufferNestedTensor> _data;
};

} // namespace nested_tensor
} // namespace torch
