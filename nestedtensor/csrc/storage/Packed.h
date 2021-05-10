#pragma once
#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {

struct PackedStorage {
  explicit PackedStorage(
      at::Tensor&& buffer,
      SizeNode nested_size,
      SizeNode nested_stride)
      : _buffer(buffer),
        _nested_size(nested_size),
        _nested_stride(nested_stride),
        _opt_sizes(construct_size(_nested_size)),
        _data_type(buffer.dtype()),
        _device(buffer.device()),
        _dim(
            get_first_leaf(_nested_size)
                ? get_first_leaf(_nested_size)->size() + _nested_size.height()
                : _nested_size.height()),
        _is_pinned(buffer.is_pinned()) {
    TORCH_CHECK(
        !_nested_size.is_leaf(),
        "PackedStorage must be given NestedSize of at least height 1.");
    TORCH_CHECK(
        !_nested_stride.is_leaf(),
        "PackedStorage must be given NestedStride of at least height 1.");
  }
  explicit PackedStorage(at::Tensor&& buffer, SizeNode nested_size)
      : PackedStorage(
            std::move(buffer),
            nested_size,
            map(
                [](std::vector<int64_t> sizes) {
                  return torch::nested_tensor::impl::_cont_stride(sizes);
                },
                nested_size)) {}
  int64_t dim() const {
    return _dim;
  }
  at::Tensor& get_buffer() {
    return _buffer;
  }
  const at::Tensor& get_buffer() const {
    return _buffer;
  }
  const caffe2::TypeMeta dtype() const {
    return _data_type;
  }
  c10::Device device() const {
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
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    return construct_size(_nested_size);
  }

 private:
  at::Tensor _buffer;
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
