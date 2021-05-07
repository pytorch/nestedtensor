#pragma once
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<std::vector<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

static std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
}

struct PackedStorage {
  explicit PackedStorage(TensorNode&& structure) :
    _structure(structure),
    _nested_size(map(
        [](at::Tensor tensor) { return tensor.sizes().vec(); },
        _structure)),
    _nested_stride(map(
        [](at::Tensor tensor) { return tensor.strides().vec(); },
        _structure)),
    _opt_sizes(construct_size(_nested_size)),
    _data_type(get_first_leaf(structure) ? get_first_leaf(structure)->dtype()
                              : at::ones({}).dtype()),
    _device(get_first_leaf(structure) ? get_first_leaf(structure)->device()
                              : at::ones({}).device()),
    _dim(get_first_leaf(structure) ? get_first_leaf(structure)->dim() + _structure.height()
                              : _structure.height()),
    _is_pinned(get_first_leaf(structure) ? get_first_leaf(structure)->is_pinned()
                              : false)
  {
    TORCH_CHECK(
        !_structure.is_leaf(),
        "NestedTensorImpl must be given structure of at least height 1.")
  }
  int64_t dim() const {
    return _dim;
  }
  TensorNode& get_structure() {
    return _structure;
  }
  const TensorNode& get_structure() const {
    return _structure;
  }
  const caffe2::TypeMeta dtype() const {
    return _data_type;
  }
  Device device() const {
    return _device;
  }
  bool is_pinned() const {
    return _is_pinned;
  }
  const SizeNode nested_size() const {
    return _nested_size;
  }
  const SizeNode nested_stride() const {
    return _nested_stride;
  }

private:
  TensorNode _structure;
  const SizeNode _nested_size;
  const SizeNode _nested_stride;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
  const caffe2::TypeMeta _data_type;
  Device _device;
  int64_t _dim;
  bool _is_pinned;
};

}
}
