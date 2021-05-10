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
      _list_storage(ListStorage(std::move(structure))),
      _kind(NestedTensorStorageKind::list) {
  }
  explicit NestedTensorStorage(ListStorage&& list_storage) :
      _list_storage(list_storage),
      _kind(NestedTensorStorageKind::list) {
  }
  int64_t dim() const {
    switch(_kind)
    {
      case packed: return _packed_storage->dim(); break;
      case list: return _list_storage->dim(); break;
    }
  }
  TensorNode& get_structure() {
    switch(_kind)
    {
      case packed: 
        TORCH_CHECK(false, "PackedStorage doesn't provide structure.");
        break;
      case list: return _list_storage->get_structure(); break;
    }
  }
  const TensorNode& get_structure() const {
    switch(_kind)
    {
      case packed: 
        TORCH_CHECK(false, "PackedStorage doesn't provide structure.");
        break;
      case list: return _list_storage->get_structure(); break;
    }
  }
  const caffe2::TypeMeta dtype() const {
    switch(_kind)
    {
      case packed: return _packed_storage->dtype(); break;
      case list: return _list_storage->dtype(); break;
    }
  }
  Device device() const {
    switch(_kind)
    {
      case packed: return _packed_storage->device(); break;
      case list: return _list_storage->device(); break;
    }
  }
  bool is_pinned() const {
    switch(_kind)
    {
      case packed: return _packed_storage->is_pinned(); break;
      case list: return _list_storage->is_pinned(); break;
    }
  }
  const SizeNode nested_size() const {
    switch(_kind)
    {
      case packed: return _packed_storage->nested_size(); break;
      case list: return _list_storage->nested_size(); break;
    }
  }
  const SizeNode nested_stride() const {
    switch(_kind)
    {
      case packed: return _packed_storage->nested_stride(); break;
      case list: return _list_storage->nested_stride(); break;
    }
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    switch(_kind)
    {
      case packed: return _packed_storage->opt_sizes(); break;
      case list: return _list_storage->opt_sizes(); break;
    }
  }

private:
  c10::optional<PackedStorage> _packed_storage;
  c10::optional<ListStorage> _list_storage;
  NestedTensorStorageKind _kind;

};
}
}
