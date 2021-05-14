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

namespace impl {
static at::Tensor stack_sizes(SizeNode size_node) {
  std::vector<at::Tensor> flattened = flatten(map(
      [](std::vector<int64_t> sizes) {
        return torch::tensor(sizes, torch::kInt64);
      },
      size_node));
  if (flattened.size() == 0) {
    return torch::zeros({0, 0}, torch::kInt64);
  }
  return at::stack(flattened);
}
} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(SizeNode size_node)
      : _height(size_node.height()),
        _opt_sizes(construct_size(size_node)),
        _structure(serialize(map(
            [](std::vector<int64_t> sizes) {
              std::vector<int64_t> result;
              return result;
            },
            size_node))),
        _sizes(impl::stack_sizes(size_node)) {}

  SizeNode to_size_node() const {
    std::vector<std::vector<int64_t>> _tmp_sizes(_sizes.size(0));
    int64_t* _sizes_ptr = _sizes.data_ptr<int64_t>();
    for (int64_t i = 0; i < _sizes.size(0); i++) {
      _tmp_sizes[i].resize(_sizes.size(1));
      for (int64_t j = 0; j < _sizes.size(1); j++) {
        _tmp_sizes[i][j] = _sizes_ptr[i * _sizes.size(1) + j];
      }
    }
    return unflatten(deserialize_size_node(_structure), _tmp_sizes);
  }
  int64_t height() const {
    return _height;
  }
  int64_t dim() const {
    return _sizes.size(0) > 0 ? _height + _sizes.size(1) : _height;
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    return _opt_sizes;
  }

 private:
  int64_t _height;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
  std::vector<int64_t> _structure;
  at::Tensor _sizes;
};

} // namespace nested_tensor
} // namespace torch
