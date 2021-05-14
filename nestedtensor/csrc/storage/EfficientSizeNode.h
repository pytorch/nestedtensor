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

static void __construct_levels(
    const SizeNode& size_node,
    int64_t level,
    std::vector<int64_t>& offsets,
    std::vector<bool>& is_vector,
    std::vector<int64_t*>& levels) {
  if (is_vector[level]) {
    levels[level][offsets[level]] = (int64_t)(size_node.degree());
    offsets[level]++;
  }
  for (const auto& child : size_node.unbind()) {
    __construct_levels(child, level + 1, offsets, is_vector, levels);
  }
}

static std::vector<at::Tensor> _construct_levels(
    const SizeNode& size_node,
    const std::vector<c10::optional<int64_t>>& opt_sizes,
    int64_t dim,
    std::vector<int64_t>& offset,
    std::vector<bool>& is_vector) {
  std::vector<int64_t> child_count(dim, 0);
  _count_children(size_node, child_count, 0);
  std::vector<at::Tensor> result;
  for (int64_t i = 0; i < dim; i++) {
    if (opt_sizes[i]) {
      result.push_back(torch::tensor({*opt_sizes[i]}, torch::kInt64));
    } else {
      result.push_back(torch::zeros({child_count[i]}, torch::kInt64));
    }
  }
  std::vector<int64_t> offsets(dim, 0);
  std::vector<bool> is_vector(dim, false);
  std::vector<int64_t*> result_ptr(dim);
  for (int64_t i = 0; i < dim; i++) {
    result_ptr[i] = result[i].data_ptr<int64_t>();
    is_vector[i] = true;
  }
  __construct_levels(size_node, 0, offsets, is_vector, result_ptr);
  return result;
}

SizeNode _construct_size_node(
    int64_t height,
    int64_t dim,
    std::vector<at::Tensor> levels,
    int64_t level,
    std::vector<int64_t>& offsets,
    std::vector<bool>& is_vector) {
  std::vector<int64_t> size;
  TORCH_CHECK(level <= height, "internal error.");
  if (level == height) {
    for (size_t i = level; i < dim; i++) {
      levels[i][offsets[i]].item<int64_t>();
    }
  }
  if (is_vector[level]) {
    std::vector<SizeNode> nodes;
    for (size_t i = 0; i < levels[level][offsets[level]]; i++) {
      nodes.push_back(
          _construct_size_node(height, levels, level + 1, offsets, is_vector));
    }
    return SizeNode(std::move(nodes));
  } else {
    std::vector<SizeNode> nodes;
    for (size_t i = 0; i < levels[level].item<int64_t>(); i++) {
      nodes.push_back(
          _construct_size_node(height, levels, level + 1, offsets, is_vector));
    }
    return SizeNode(std::move(nodes));
  }
  return levels[0];
}

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(
      const SizeNode& size_node,
      const std::vector<c10::optional<int64_t>>& opt_sizes,
      int64_t dim)
      : _height(size_node.height()),
        _dim(dim),
        _offsets(dim, 0),
        _is_vector(dim, false),
        _levels(impl::_construct_levels(
            size_node,
            opt_sizes,
            dim,
            _offsets,
            _is_vector)),
        _opt_sizes(opt_sizes) {}

  SizeNode to_size_node() {
    for (size_t i = 0; i < _offsets.size(); i++) {
      _offsets[i] = 0;
    }
    return impl::_construct_size_node(_height, _levels, 0, _offsets, _levels);
  }

 private:
  int64_t _height;
  int64_t _dim;
  std::vector<int64_t> _offsets;
  std::vector<bool> _is_vector;
  std::vector<at::Tensor> _levels;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
};

} // namespace nested_tensor
} // namespace torch
