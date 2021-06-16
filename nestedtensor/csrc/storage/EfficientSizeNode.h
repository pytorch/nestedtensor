#pragma once
#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {

namespace impl {
inline at::Tensor stack_sizes(SizeNode size_node) {
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

inline std::vector<c10::optional<int64_t>> construct_efficient_size(
    int64_t out,
    int64_t height,
    const at::Tensor& sizes) {
  std::vector<c10::optional<int64_t>> result;
  result.push_back(out);
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

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(const SizeNode& size_node)
      : _height(size_node.height()),
        _structure(size_node.degree()),
        _sizes(impl::stack_sizes(size_node)),
        _opt_sizes(impl::construct_efficient_size(_structure, _height, _sizes))
  {}

  explicit EfficientSizeNode(
      int64_t height,
      int64_t structure,
      const at::Tensor& sizes)
      : _height(height),
        _structure(structure), 
        _sizes(sizes),
        _opt_sizes(impl::construct_efficient_size(_structure, _height, _sizes))
  {}

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
    std::vector<SizeNode> _tmp_size_nodes;
    for (int64_t i = 0; i < _structure; i++) {
      _tmp_size_nodes.push_back(SizeNode(std::move(_tmp_sizes[i])));
    }
    return SizeNode(std::move(_tmp_size_nodes));
  }
  int64_t height() const {
    return _height;
  }
  int64_t dim() const {
    return _sizes.dim() > 0 ? _height + _sizes.size(1) : _height;
  }
  const std::vector<c10::optional<int64_t>>& opt_sizes() const {
    return _opt_sizes;
  }
  void refresh_opt_sizes() {
    _opt_sizes = impl::construct_efficient_size(_structure, _height, _sizes);
  }
  const at::Tensor& sizes() const {
    return _sizes;
  }
  const int64_t structure() const {
    return _structure;
  }
  EfficientSizeNode clone() const {
    return EfficientSizeNode(_height, _structure, _sizes.clone());
  }
  int64_t numel() const {
    if (_sizes.dim() == 0 && _structure > 0) {
      return _structure;
    }
    if (_sizes.dim() > 0) {
      if (_sizes.numel() == 0) {
        return 0;
      }
      Tensor nt_sizes = at::native::narrow(
          _sizes, 1 /* dim */, 0 /* start */, 1 /* length */);
      for (int64_t i = 1; i < _sizes.size(1); i++) {
        Tensor tmp = at::native::narrow(
            _sizes, 1 /* dim */, i /* start */, 1 /* length */);
        nt_sizes = nt_sizes * tmp;
      }
      return nt_sizes.sum().item<int64_t>();
    }
    return 0;
  }

 private:
  int64_t _height;
  int64_t _structure;
  const at::Tensor _sizes;
  bool _opt_sizes_set = false;
  std::vector<c10::optional<int64_t>> _opt_sizes;
};

inline bool efficient_size_structure_matches(
    EfficientSizeNode& size_node0,
    EfficientSizeNode& size_node1) {
  return size_node0.structure() == size_node1.structure();
}

inline bool efficient_size_matches(
    EfficientSizeNode& size_node0,
    EfficientSizeNode& size_node1) {
  if (!efficient_size_structure_matches(size_node0, size_node1)) {
    return false;
  }
  at::Tensor sizes0 = size_node0.sizes();
  at::Tensor sizes1 = size_node1.sizes();
  return at::equal(sizes0, sizes1);
}

template <class F>
inline EfficientSizeNode map_efficient_size(
    F&& fn,
    const EfficientSizeNode& size_node) {
  at::Tensor sizes = size_node.sizes().clone();
  if (sizes.dim() == 0) {
    return EfficientSizeNode(size_node.height(), size_node.structure(), sizes);
  }
  int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes.size(0); i++) {
    fn(sizes_ptr + i * sizes.size(1), sizes.size(1));
  }
  return EfficientSizeNode(size_node.height(), size_node.structure(), sizes);
}

template <class F>
inline void apply_efficient_size(
    F&& fn,
    EfficientSizeNode& size_node0,
    EfficientSizeNode& size_node1) {
  at::Tensor sizes0 = size_node0.sizes();
  at::Tensor sizes1 = size_node1.sizes();
  int64_t* sizes0_ptr = sizes0.data_ptr<int64_t>();
  int64_t* sizes1_ptr = sizes1.data_ptr<int64_t>();
  int64_t structure0 = size_node0.structure();
  int64_t structure1 = size_node1.structure();
  TORCH_CHECK(
      efficient_size_structure_matches(size_node0, size_node1),
      "apply_efficient_size: Length doesn't match.");
  for (int64_t i = 0; i < sizes0.size(0); i++) {
    fn(sizes0_ptr + i * sizes0.size(1),
       sizes0.size(1),
       sizes1_ptr + i * sizes1.size(1),
       sizes1.size(1));
  }
  size_node0.refresh_opt_sizes();
  size_node1.refresh_opt_sizes();
}

} // namespace nested_tensor
} // namespace torch
