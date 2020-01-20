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
std::vector<py::object> unbind_THPSizeNode(
    SizeNode size_node,
    std::string name);

struct THPSizeNode {
  THPSizeNode(SizeNode size_node, std::string name)
      : _size_node(size_node),
        _name(name),
        _elements(unbind_THPSizeNode(_size_node, _name)) {}
  int64_t len() {
    if (_size_node.is_leaf()) {
      return _size_node.size();
    } else {
      return _size_node.degree();
    }
  }
  std::string str() {
    return SizeNode___str__(_size_node, _name);
  }
  const SizeNode& get_size_node() {
    return _size_node;
  }
  std::string get_name() {
    return _name;
  }
  const std::vector<py::object>& get_elements() {
    return _elements;
  }

 private:
  SizeNode _size_node;
  std::string _name;
  std::vector<py::object> _elements;
};

template <class Result, class F>
static inline Result data_map(
    c10::either<_ListNestedTensor, _BufferNestedTensor>& data,
    F fn) {
  return data.map<Result>(fn, fn);
}

static inline std::vector<c10::optional<int64_t>> _construct_size(
    const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    result.push_back(size_node.size());
    if (size_node.size() == 0) {
      return result;
    }

    for (const auto& size : size_node.payload(0)) {
      result.push_back(size);
    }

    for (size_t j = 1; j < result.size(); j++) {
      for (size_t i = 1; i < size_node.size(); i++) {
        if (!result[j]) {
          break;
        }
        if ((*(result[j])) != size_node.payload(i)[j - 1]) {
          result[j] = c10::nullopt;
        }
      }
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : _construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = _construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
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
    return _construct_size(this->nested_size().get_size_node());
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
  THPNestedTensor requires_grad_(pybind11::bool_ requires_grad) {
    return THPNestedTensor(
        data_map<THPNestedTensor>(_data, [&requires_grad](auto data) {
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
      return TensorNode___str__(data.get_structure());
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
