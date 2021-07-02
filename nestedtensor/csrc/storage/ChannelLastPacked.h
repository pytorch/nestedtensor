#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>
#include <nestedtensor/csrc/utils/nested_node.h>

namespace torch {
namespace nested_tensor {

struct ChannelLastPackedStorage : public NestedTensorStorage {
  explicit ChannelLastPackedStorage(
      at::Tensor&& buffer,
      EfficientSizeNode nested_size)
      : _buffer(buffer),
        _nested_size(nested_size),
        _data_type(buffer.dtype()),
        _device(buffer.device()),
        _is_pinned(buffer.is_pinned()) {
    TORCH_CHECK(
        _nested_size.height() == 1,
        "PackedStorage must be given NestedSize of exactly height 1.");
  }

  int64_t dim() const override {
    return _nested_size.dim();
  }
  at::Tensor& get_buffer() {
    return _buffer;
  }
  const at::Tensor& get_buffer() const {
    return _buffer;
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
  const EfficientSizeNode& nested_size() const override {
    return _nested_size;
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const override {
    return _nested_size.opt_sizes();
  }
  NestedTensorStorageKind kind() const override {
    return NestedTensorStorageKind::channellastpacked;
  }
  bool is_cuda() const override {
    return _buffer.is_cuda();
  }
  int64_t numel() const override {
    return _nested_size.numel();
  }

 private:
  at::Tensor _buffer;
  EfficientSizeNode _nested_size;
  const caffe2::TypeMeta _data_type;
  c10::Device _device;
  bool _is_pinned;
};

} // namespace nested_tensor
} // namespace torch
