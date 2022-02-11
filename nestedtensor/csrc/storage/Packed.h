#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <c10/core/MemoryFormat.h>

namespace torch {
namespace nested_tensor {
namespace impl {

inline EfficientSizeNode _cont_stride(const EfficientSizeNode& nested_size) {
  auto nested_stride = map_efficient_size(
      [](int64_t* size_ptr, int64_t size) {
        auto cont_stride = _cont_stride(size_ptr, size);
        for (int64_t i = 0; i < size; i++) {
          size_ptr[i] = cont_stride[i];
        }
      }, nested_size);
  return nested_stride;
}

inline std::tuple<TensorNode, at::Tensor> build_structure(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size_,
    const EfficientSizeNode& nested_stride_) {
  TORCH_CHECK(
      buffer.dim() == 1, "Given buffer must be vector, i.e. dim 1 Tensor.");
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(nested_size_.degree());
  map_efficient_size([&split_sizes] (int64_t* sizes_ptr0, int64_t* sizes_ptr1, int64_t size) {
      split_sizes.push_back(num_memory(sizes_ptr0, sizes_ptr1, size));
      }, nested_size_, nested_stride_);
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
  std::vector<TensorNode> result_tensors;
  index = 0;
  map_efficient_size([&buffers, &result_tensors, &index](
        int64_t* size_ptr, int64_t* stride_ptr, int64_t size) {
      std::vector<int64_t> sizes(size_ptr, size_ptr + size);
      std::vector<int64_t> strides(stride_ptr, stride_ptr + size);
      result_tensors.push_back(TensorNode(at::as_strided(
            buffers[index], c10::IntArrayRef(sizes), c10::IntArrayRef(strides))));
      index++;
      }, nested_size_, nested_stride_);
  return std::make_tuple(TensorNode(std::move(result_tensors)), buffer);
}

inline std::tuple<TensorNode, at::Tensor> build_structure(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size) {
  TORCH_CHECK(
      buffer.dim() == 1, "Given buffer must be vector, i.e. dim 1 Tensor.");
  EfficientSizeNode nested_stride = _cont_stride(nested_size);
  return build_structure(buffer, nested_size, nested_stride);
}

inline at::Tensor pack(const TensorNode& structure) {
  TORCH_CHECK(structure.height() == 1, "Expected structure of height 1, got ", structure.height(), " instead.");
  if (structure.degree() == 0) {
    return at::ones({0});
  }
  auto tensor_nodes = structure.unbind();
  std::vector<at::Tensor> tensors;
  tensors.resize(structure.degree());
  int64_t full_numel = 0;
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors[i] = tensor_nodes[i].payload();
    full_numel = full_numel + tensors[i].numel();
  }
  at::Tensor result_buffer = empty({full_numel}, tensors[0].options());
  int64_t index = 0;
  for (size_t i = 0; i < tensors.size(); i++) {
    at::Tensor narrowed_result_buffer = 
      result_buffer.narrow(0, index, tensors[i].numel());
    narrowed_result_buffer = narrowed_result_buffer.reshape(tensors[i].sizes());
    narrowed_result_buffer.copy_(tensors[i], true);
    index = index + tensors[i].numel();
  }
  return result_buffer;
}

inline bool storage_is_contiguous(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size,
    const EfficientSizeNode& nested_stride) {
  if (!buffer.is_contiguous()) {
    return false;
  }
  if (buffer.numel() == 0) {
    return true;
  }
  const at::Tensor& sizes_sizes = nested_size.sizes();
  const at::Tensor& strides_sizes = nested_stride.sizes();
  int64_t* sizes_sizes_ptr = sizes_sizes.data_ptr<int64_t>();
  int64_t* strides_sizes_ptr = strides_sizes.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes_sizes.size(0); i++) {
    if (!_is_cont_stride(
            sizes_sizes_ptr + i * sizes_sizes.size(1),
            strides_sizes_ptr + i * strides_sizes.size(1),
            sizes_sizes.size(1))) {
      return false;
    }
  }
  return true;
}

inline bool storage_is_contiguous_channels_last(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size,
    const EfficientSizeNode& nested_stride) {
  if (!buffer.is_contiguous()) {
    return false;
  }
  if (buffer.numel() == 0) {
    return true;
  }
  if (nested_size.dim() != 4) {
    return false;
  }
  const at::Tensor& sizes_sizes = nested_size.sizes();
  const at::Tensor& strides_sizes = nested_stride.sizes();
  int64_t* sizes_sizes_ptr = sizes_sizes.data_ptr<int64_t>();
  int64_t* strides_sizes_ptr = strides_sizes.data_ptr<int64_t>();
  std::vector<int64_t> sizes(4, 0);
  std::vector<int64_t> strides(4, 0);
  for (int64_t i = 0; i < sizes_sizes.size(0); i++) {
    sizes[0] = 1;
    sizes[1] = sizes_sizes_ptr[i * 3 + 0];
    sizes[2] = sizes_sizes_ptr[i * 3 + 1];
    sizes[3] = sizes_sizes_ptr[i * 3 + 2];
    strides[0] = sizes_sizes_ptr[i * 3 + 0] *
                 sizes_sizes_ptr[i * 3 + 1] *
                 sizes_sizes_ptr[i * 3 + 2];
    strides[1] = strides_sizes_ptr[i * 3 + 0];
    strides[2] = strides_sizes_ptr[i * 3 + 1];
    strides[3] = strides_sizes_ptr[i * 3 + 2];
    if (!c10::is_channels_last_strides_2d(IntArrayRef(sizes), IntArrayRef(strides))) {
      return false;
    }
  }
  return true;
}

} // namespace impl
} // namespace nested_tensor
} // namespace torch
