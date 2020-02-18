#include <list_nested_tensor.h>

namespace torch {
namespace nested_tensor {

c10::List<int64_t> _cont_stride(c10::List<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return c10::List<int64_t>(stride);
}

TensorNode build_structure(
    at::Tensor buffer,
    SizeNode nested_size,
    SizeNode nested_stride) {
  c10::List<int64_t> split_sizes = flatten(
      map([](c10::List<int64_t> a,
             c10::List<int64_t> b) { return num_memory(a, b); },
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
  TensorNode tmp = unflatten(nested_size, c10::List<at::Tensor>(buffers));
  return map(
      [](at::Tensor buffer,
         c10::List<int64_t> size,
         c10::List<int64_t> stride) {
        return at::as_strided(
            buffer,
            c10::IntArrayRef(size.vec()),
            c10::IntArrayRef(stride.vec()));
      },
      tmp,
      nested_size,
      nested_stride);
}

NestedTensor NestedTensor::contiguous() {
  if (this->is_contiguous()) {
    return *this;
  }
  auto node = _data.get_structure();
  c10::List<at::Tensor> tensors;
  for (const at::Tensor& tensor : flatten(_structure)) {
    tensors.emplace_back(tensor.reshape({-1}));
  }
  at::Tensor buffer;
  if (tensors.size() == 0) {
    buffer = torch::ones({});
  } else {
    buffer = at::cat(tensors.vec(), 0);
  }
  SizeNode nested_size =
      map([](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
          _structure);
  SizeNode nested_stride = map(
      [](c10::List<int64_t> size) { return _cont_stride(size); }, nested_size);
  _structure = build_structure(buffer, nested_size, nested_stride);
  _first_variable =
      get_first_leaf(_structure) ? *get_first_leaf(_structure) : at::ones({});
  return *this;
}

} // namespace nested_tensor
} // namespace torch
