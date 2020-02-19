#include <nested_tensor.h>

namespace torch {
namespace nested_tensor {

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
}

std::vector<c10::optional<int64_t>> NestedTensor::size() {
  return construct_size(_nested_size);
}

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
    const at::Tensor& buffer,
    const SizeNode& nested_size,
    const SizeNode& nested_stride) {
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

TensorNode build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size) {
  SizeNode nested_stride = map(
      [](c10::List<int64_t> size) { return _cont_stride(size); }, nested_size);
  return build_structure(buffer, nested_size, nested_stride);
}

SizeNode infer_nested_size(const TensorNode& _structure) {
  return map(
      [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
      _structure);
}

NestedTensor NestedTensor::contiguous() {
  if (is_contiguous()) {
    return *this;
  }
  c10::List<at::Tensor> tensors;
  for (const at::Tensor& tensor : flatten(_structure)) {
    tensors.emplace_back(tensor.reshape({-1}));
  }
  at::Tensor buffer;
  if (tensors.size() == 0) {
    buffer = at::ones({});
  } else {
    buffer = at::cat(tensors.vec(), 0);
  }
  _structure = build_structure(buffer, _nested_size);
  _first_variable =
      get_first_leaf(_structure) ? *get_first_leaf(_structure) : at::ones({});
  return *this;
}

NestedTensor::NestedTensor(TensorNode&& structure)
    : _structure(structure),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(infer_nested_size(_structure)) {}

NestedTensor::NestedTensor(at::Tensor&& buffer, SizeNode nested_size)
    : _buffer(buffer),
      _structure(build_structure(buffer, nested_size)),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(infer_nested_size(_structure)) {}

} // namespace nested_tensor
} // namespace torch
