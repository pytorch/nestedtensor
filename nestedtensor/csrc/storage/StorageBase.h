#pragma once
#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {

enum NestedTensorStorageKind { packed, list };

struct NestedTensorStorage {
  virtual ~NestedTensorStorage() = default;
  virtual int64_t dim() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual TensorNode get_structure() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const caffe2::TypeMeta dtype() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual c10::Device device() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual bool is_pinned() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const EfficientSizeNode& nested_size() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const EfficientSizeNode& nested_stride() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const std::vector<c10::optional<int64_t>> opt_sizes() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual NestedTensorStorageKind kind() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual bool is_contiguous(at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual bool is_cuda() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual int64_t numel() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
};
} // namespace nested_tensor
} // namespace torch
