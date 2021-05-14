#pragma once
#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {
// namespace impl {
//
// static void _count_children(
//     const SizeNode& size_node,
//     std::vector<int64_t>& child_count,
//     int64_t level) {
//   child_count[level] += size_node.degree();
//   for (auto const& child : size_node.unbind()) {
//     _count_children(child, child_count, level + 1);
//   }
// }
//
// } // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(
      SizeNode size_node,
      const std::vector<c10::optional<int64_t>>& opt_sizes)
      : _height(size_node.height()),
        _opt_sizes(opt_sizes),
        _structure(serialize(map(
            [](std::vector<int64_t> sizes) {
              std::vector<int64_t> result;
              return result;
            },
            size_node))),
        _sizes(flatten(size_node)) {}

  SizeNode to_size_node() const {
    return unflatten(deserialize_size_node(_structure), _sizes);
  }
  int64_t height() const {
    return _height;
  }
  int64_t dim() const {
    return _sizes.size() > 0 ? _height + _sizes[0].size() : _height;
  }

 private:
  int64_t _height;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
  std::vector<int64_t> _structure;
  std::vector<std::vector<int64_t>> _sizes;
};

} // namespace nested_tensor
} // namespace torch
