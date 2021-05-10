#pragma once
#include <nestedtensor/csrc/storage/StorageBase.h>

namespace torch {
namespace nested_tensor {

struct ListStorage : public NestedTensorStorage {
  explicit ListStorage(TensorNode&& structure)
      : _kind(NestedTensorStorageKind::list),
        _structure(structure),
        _nested_size(
            map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                _structure)),
        _nested_stride(
            map([](at::Tensor tensor) { return tensor.strides().vec(); },
                _structure)),
        _opt_sizes(construct_size(_nested_size)),
        _data_type(
            get_first_leaf(structure) ? get_first_leaf(structure)->dtype()
                                      : at::ones({}).dtype()),
        _device(
            get_first_leaf(structure) ? get_first_leaf(structure)->device()
                                      : at::ones({}).device()),
        _dim(
            get_first_leaf(structure)
                ? get_first_leaf(structure)->dim() + _structure.height()
                : _structure.height()),
        _is_pinned(
            get_first_leaf(structure) ? get_first_leaf(structure)->is_pinned()
                                      : false) {
    TORCH_CHECK(
        !_structure.is_leaf(),
        "NestedTensorImpl must be given structure of at least height 1.");
  }
  int64_t dim() const override {
    return _dim;
  }
  TensorNode& get_structure() override {
    return _structure;
  }
  const TensorNode& get_structure() const override {
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
  const SizeNode nested_size() const override {
    return _nested_size;
  }
  const SizeNode nested_stride() const override {
    return _nested_stride;
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const override {
    return _opt_sizes;
  }

 private:
  NestedTensorStorageKind _kind;
  TensorNode _structure;
  const SizeNode _nested_size;
  const SizeNode _nested_stride;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
  const caffe2::TypeMeta _data_type;
  c10::Device _device;
  int64_t _dim;
  bool _is_pinned;
};

} // namespace nested_tensor
} // namespace torch
