#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>

namespace torch {
namespace nested_tensor {

struct ListStorage : public NestedTensorStorage {
  explicit ListStorage(TensorNode&& structure)
      : _structure(structure),
        _nested_size(EfficientSizeNode(
            map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                _structure))),
        _nested_stride(EfficientSizeNode(
            map([](at::Tensor tensor) { return tensor.strides().vec(); },
                _structure))),
        _data_type(
            get_first_leaf(_structure) ? get_first_leaf(_structure)->dtype()
                                       : at::ones({}).dtype()),
        _device(
            get_first_leaf(_structure) ? get_first_leaf(_structure)->device()
                                       : at::ones({}).device()),
        _is_pinned(
            get_first_leaf(_structure) ? get_first_leaf(_structure)->is_pinned()
                                       : false) {
    TORCH_CHECK(
        !_structure.is_leaf(),
        "NestedTensorImpl must be given structure of at least height 1.");
  }
  int64_t dim() const override {
    return _nested_size.dim();
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
  EfficientSizeNode nested_size() const override {
    return _nested_size;
  }
  EfficientSizeNode nested_stride() const override {
    return _nested_stride;
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const override {
    return _nested_size.opt_sizes();
  }
  NestedTensorStorageKind kind() const override {
    return NestedTensorStorageKind::list;
  }
  bool is_contiguous() const override {
    return false;
  }
  bool is_cuda() const override {
    return get_first_leaf(_structure) ? get_first_leaf(_structure)->is_cuda()
                                      : false;
  }

 private:
  TensorNode _structure;
  EfficientSizeNode _nested_size;
  EfficientSizeNode _nested_stride;
  const caffe2::TypeMeta _data_type;
  c10::Device _device;
  bool _is_pinned;
}; // namespace nested_tensor

} // namespace nested_tensor
} // namespace torch
