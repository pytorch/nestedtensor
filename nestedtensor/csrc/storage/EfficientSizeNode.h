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
    int64_t dim) {
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

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(
      const SizeNode& size_node,
      const std::vector<c10::optional<int64_t>>& opt_sizes,
      int64_t dim)
      : _height(size_node.height()),
        _levels(impl::_construct_levels(size_node, opt_sizes, dim)),
        _opt_sizes(opt_sizes) {}

 private:
  int64_t _height;
  std::vector<at::Tensor> _levels;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
};

} // namespace nested_tensor
} // namespace torch
