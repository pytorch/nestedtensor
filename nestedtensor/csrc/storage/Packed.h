#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>

namespace torch {
namespace nested_tensor {
namespace impl {
inline std::tuple<TensorNode, at::Tensor> build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size,
    const SizeNode& nested_stride) {
  std::vector<int64_t> split_sizes = flatten(
      map([](std::vector<int64_t> a,
             std::vector<int64_t> b) { return num_memory(a, b); },
          nested_size,
          nested_stride));
  std::vector<int64_t> nonzero_split_sizes;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      nonzero_split_sizes.push_back(split_sizes[i]);
    }
  }
  std::vector<at::Tensor> buffers_;
  if (nonzero_split_sizes.size() > 0) {
    buffers_ =
        at::split_with_sizes(buffer, c10::IntArrayRef(nonzero_split_sizes), 0);
  }
  std::vector<at::Tensor> buffers;
  int64_t index = 0;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      buffers.push_back(buffers_[index]);
      index++;
    } else {
      buffers.push_back(at::empty({}, buffer.options()));
    }
  }
  TensorNode tmp = unflatten(nested_size, std::move(buffers));
  TensorNode result = map(
      [](at::Tensor buffer,
         std::vector<int64_t> size,
         std::vector<int64_t> stride) {
        return at::as_strided(
            buffer, c10::IntArrayRef(size), c10::IntArrayRef(stride));
      },
      tmp,
      nested_size,
      nested_stride);
  return std::make_tuple(result, buffer);
}

inline std::tuple<TensorNode, at::Tensor> build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size) {
  TORCH_CHECK(
      buffer.dim() == 1, "Given buffer must be vector, i.e. dim 1 Tensor.");
  SizeNode nested_stride =
      map([](std::vector<int64_t> size) { return _cont_stride(size); },
          nested_size);
  return build_structure(buffer, nested_size, nested_stride);
}

inline at::Tensor pack(const TensorNode& structure) {
  TensorNode flat_structure =
      map([](at::Tensor tensor) { return tensor.reshape({-1}); }, structure);
  auto nested_size =
      map([](at::Tensor tensor) { return tensor.sizes().vec(); }, structure);
  auto tensors = flatten(flat_structure);
  if (tensors.size() == 0) {
    return std::get<1>(impl::build_structure(at::ones({0}), nested_size));
  }
  return std::get<1>(impl::build_structure(at::cat(tensors, 0), nested_size));
}
} // namespace impl

struct PackedStorage : public NestedTensorStorage {
  explicit PackedStorage(
      at::Tensor&& buffer,
      SizeNode nested_size,
      SizeNode nested_stride)
      : _buffer(buffer),
        _opt_sizes(construct_size(nested_size)),
        _dim(
            get_first_leaf(nested_size)
                ? get_first_leaf(nested_size)->size() + nested_size.height()
                : nested_size.height()),
        _nested_size(EfficientSizeNode(nested_size, _opt_sizes, _dim)),
        _nested_stride(EfficientSizeNode(nested_stride, _opt_sizes, _dim)),
        _data_type(buffer.dtype()),
        _device(buffer.device()),
        _is_pinned(buffer.is_pinned()) {
    TORCH_CHECK(
        _nested_size.height(),
        "PackedStorage must be given NestedSize of at least height 1.");
    TORCH_CHECK(
        _nested_stride.height(),
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
  explicit PackedStorage(TensorNode structure)
      : PackedStorage(
            impl::pack(structure),
            map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                structure)) {}

  int64_t dim() const override {
    return _dim;
  }
  TensorNode get_structure() const {
    return std::get<0>(impl::build_structure(
        _buffer, _nested_size.to_size_node(), _nested_stride.to_size_node()));
  }
  at::Tensor& get_buffer() {
    return _buffer;
  }
  const at::Tensor& get_buffer() const {
    return _buffer;
  }
  const caffe2::TypeMeta dtype() const override {
    return _data_type;
  }
  c10::Device device() const override {
    return _device;
  }
  bool is_pinned() const override {
    return _is_pinned;
  }
  SizeNode nested_size() const override {
    return _nested_size.to_size_node();
  }
  SizeNode nested_stride() const override {
    return _nested_stride.to_size_node();
  }
  const std::vector<c10::optional<int64_t>>& opt_sizes() const override {
    return _opt_sizes;
  }
  NestedTensorStorageKind kind() const {
    return NestedTensorStorageKind::packed;
  }
  bool is_contiguous() const {
    return _buffer.is_contiguous() &&
        reduce(
               [](std::vector<int64_t> sizes,
                  std::vector<int64_t> strides,
                  bool input) {
                 std::vector<int64_t> cont_strides = impl::_cont_stride(sizes);
                 bool equal = true;
                 if (sizes.size() != strides.size()) {
                   TORCH_CHECK(false, "Sizes and strides don't match in size.");
                 }
                 for (size_t i = 0; i < sizes.size(); i++) {
                   equal = equal && (strides[i] == cont_strides[i]);
                 }
                 return equal && input;
               },
               true,
               _nested_size.to_size_node(),
               _nested_stride.to_size_node());
  }

 private:
  at::Tensor _buffer;
  const std::vector<c10::optional<int64_t>> _opt_sizes;
  int64_t _dim;
  EfficientSizeNode _nested_size;
  EfficientSizeNode _nested_stride;
  const caffe2::TypeMeta _data_type;
  c10::Device _device;
  bool _is_pinned;
};

} // namespace nested_tensor
} // namespace torch
