#pragma once
#include <nestedtensor/csrc/storage/Packed.h>
#include <nestedtensor/csrc/storage/List.h>

namespace torch {
namespace nested_tensor {

enum NestedTensorStorageKind {
    packed,
    list
};

struct NestedTensorStorage {
  explicit NestedTensorStorage(PackedStorage&& packed_storage) :
      _packed_storage(packed_storage),
      _kind(NestedTensorStorageKind::packed) {
  }
  explicit NestedTensorStorage(TensorNode&& structure) :
      _packed_storage(PackedStorage(std::move(structure))),
      _kind(NestedTensorStorageKind::packed) {
  }
  explicit NestedTensorStorage(ListStorage&& list_storage) :
      _list_storage(list_storage),
      _kind(NestedTensorStorageKind::list) {
  }
  int64_t dim() const {
    return _packed_storage->dim();
  }
  TensorNode& get_structure() {
    return _packed_storage->get_structure();
  }
  const TensorNode& get_structure() const {
    return _packed_storage->get_structure();
  }
  const caffe2::TypeMeta dtype() const {
    return _packed_storage->dtype();
  }
  Device device() const {
    return _packed_storage->device();
  }
  bool is_pinned() const {
    return _packed_storage->is_pinned();
  }
  const SizeNode nested_size() const {
    return _packed_storage->nested_size();
  }
  const SizeNode nested_stride() const {
    return _packed_storage->nested_stride();
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    return _packed_storage->opt_sizes();
  }

private:
  c10::optional<PackedStorage> _packed_storage;
  c10::optional<ListStorage> _list_storage;
  NestedTensorStorageKind _kind;

};
}
}
