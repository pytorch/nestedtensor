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
    return torch::zeros({}, torch::kInt64);
  }
  return at::stack(flattened);
}
static std::vector<c10::optional<int64_t>> construct_efficient_size(
    SizeNode size_node,
    at::Tensor sizes) {
  std::vector<c10::optional<int64_t>> result = construct_size(size_node);
  size_t nested_dim = result.size();
  if (sizes.dim() > 0) {
    int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
    result.resize(nested_dim + sizes.size(1));
    for (int64_t i = 0; i < sizes.size(1); i++) {
      result[nested_dim + i] = sizes_ptr[i];
    }
    for (int64_t j = 0; j < sizes.size(1); j++) {
      for (int64_t i = 0; i < sizes.size(0); i++) {
        if (result[nested_dim + j] &&
            (result[nested_dim + j] != sizes_ptr[i * sizes.size(1) + j])) {
          result[nested_dim + j] = c10::nullopt;
        }
      }
    }
  }
  return result;
}

inline void _efficient_serialize(
    SizeNode nested_node,
    std::vector<int64_t>& out) {
  if (!nested_node.is_leaf()) {
    out.push_back(nested_node.degree());
    for (size_t i = 0; i < nested_node.degree(); i++) {
      _efficient_serialize(nested_node.children(i), out);
    }
  }
}

inline std::vector<int64_t> efficient_serialize(SizeNode nested_node) {
  std::vector<int64_t> out;
  _efficient_serialize(nested_node, out);
  return out;
}

inline std::tuple<size_t, SizeNode> _efficient_deserialize(
    std::vector<int64_t> out,
    size_t index,
    int64_t height) {
  if (height == 0) {
    return std::make_tuple(index, SizeNode(std::vector<int64_t>()));
  } else {
    int64_t degree = out[index];
    index++;
    std::vector<SizeNode> children;
    for (int64_t i = 0; i < degree; i++) {
      auto result_i = _efficient_deserialize(out, index, height - 1);
      index = std::get<0>(result_i);
      children.push_back(std::get<1>(result_i));
    }
    return std::make_tuple(index, SizeNode(std::move(children)));
  }
}

inline SizeNode efficient_deserialize(
    std::vector<int64_t> out,
    int64_t height) {
  auto tmp = _efficient_deserialize(out, 0, height);
  return std::get<1>(tmp);
}

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(SizeNode size_node)
      : _height(size_node.height()),
        _structure(impl::efficient_serialize(size_node)),
        _sizes(impl::stack_sizes(size_node)),
        _opt_sizes(impl::construct_efficient_size(
            impl::efficient_deserialize(_structure, _height),
            _sizes)) {}

  SizeNode to_size_node() const {
    std::vector<std::vector<int64_t>> _tmp_sizes;
    if (_sizes.dim() > 0) {
      _tmp_sizes.resize(_sizes.size(0));
      int64_t* _sizes_ptr = _sizes.data_ptr<int64_t>();
      for (int64_t i = 0; i < _sizes.size(0); i++) {
        _tmp_sizes[i].resize(_sizes.size(1));
        for (int64_t j = 0; j < _sizes.size(1); j++) {
          _tmp_sizes[i][j] = _sizes_ptr[i * _sizes.size(1) + j];
        }
      }
    }
    return unflatten(impl::efficient_deserialize(_structure, _height), _tmp_sizes);
  }
  int64_t height() const {
    return _height;
  }
  int64_t dim() const {
    return _sizes.dim() > 0 ? _height + _sizes.size(1) : _height;
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    return _opt_sizes;
  }
  const at::Tensor& sizes() const {
    return _sizes;
  }

 private:
  int64_t _height;
  std::vector<int64_t> _structure;
  at::Tensor _sizes;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
};

} // namespace nested_tensor
} // namespace torch
