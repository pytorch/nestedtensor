#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>

namespace torch {
namespace nested_tensor {

struct ListStorage : public NestedTensorStorage {
  explicit ListStorage(TensorNode&& structure)
      : _structure(structure),
        _nested_size(
            map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                _structure)),
        _nested_stride(
            map([](at::Tensor tensor) { return tensor.strides().vec(); },
                _structure)),
        _data_type(
            get_first_leaf(_structure) ? get_first_leaf(_structure)->dtype()
                                       : at::ones({}).dtype()),
        _device(
            get_first_leaf(_structure) ? get_first_leaf(_structure)->device()
                                       : at::ones({}).device()),
        _dim(
            get_first_leaf(_structure)
                ? get_first_leaf(_structure)->dim() + _structure.height()
                : _structure.height()),
        _is_pinned(
            get_first_leaf(_structure) ? get_first_leaf(_structure)->is_pinned()
                                       : false),
        _opt_sizes(construct_size(_nested_size)) {
    TORCH_CHECK(
        !_structure.is_leaf(),
        "NestedTensorImpl must be given structure of at least height 1.");
  }
  int64_t dim() const override {
    return _dim;
  }
  TensorNode get_structure() const override {
    return _structure;
  }
  const caffe2::TypeMeta dtype() const override {
    return _data_type;
  }
  c10::Device device() const override {
    return _device;
  }
  bool is_pinned() const override {
    return _is_pinned;
  }
  SizeNode nested_size() const override {
    return _nested_size;
  }
  SizeNode nested_stride() const override {
    return _nested_stride;
  }
  const std::vector<c10::optional<int64_t>>& opt_sizes() const override {
    return _opt_sizes;
  }
  NestedTensorStorageKind kind() const {
    return NestedTensorStorageKind::list;
  }
  bool is_contiguous() const {
    return false;
  }

 private:
  TensorNode _structure;
  const SizeNode _nested_size;
  const SizeNode _nested_stride;
  const caffe2::TypeMeta _data_type;
  c10::Device _device;
  int64_t _dim;
  bool _is_pinned;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
};

} // namespace nested_tensor
} // namespace torch
