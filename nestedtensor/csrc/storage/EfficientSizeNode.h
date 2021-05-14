#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {
namespace impl {

static void _count_children(
    const SizeNode& size_node,
    std::vector<int64_t>& child_count,
    int64_t level) {
  child_count[level] += size_node.degree();
  for (auto const& child : size_node.unbind()) {
    _count_children(child, child_count, level + 1);
  }
}

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(
      const SizeNode& size_node,
      const std::vector<c10::optional<int64_t>>& opt_sizes,
      int64_t dim)
      : _height(size_node.height()),
        _dim(dim),
        _opt_sizes(opt_sizes),
        _structure(map(
            [](std::vector<int64_t> sizes) {
              std::vector<int64_t> result;
              return result;
            },
            size_node)),
        _sizes(flatten(size_node)) {}

  SizeNode to_size_node() {
    return unflatten(_structure, _sizes);
  }

 private:
  int64_t _height;
  int64_t _dim;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
  SizeNode _structure;
  std::vector<std::vector<int64_t>> _sizes;
};

} // namespace nested_tensor
} // namespace torch
