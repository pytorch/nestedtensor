#include <nested_tensor.h>

namespace torch {
namespace nested_tensor {

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  if (size.size() == 0) {
    return 1;
  }
  return size[0] * stride[0];
}

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

NestedTensor NestedTensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  TensorNode flat_structure =
      map([](at::Tensor tensor) { return tensor.reshape({-1}); }, _structure);
  auto tensors = flatten(flat_structure).vec();
  if (tensors.size() == 0) {
    return NestedTensor(at::ones({0}), _nested_size);
  }
  return NestedTensor(at::cat(tensors, 0), _nested_size);
}

at::Tensor _to_tensor(TensorNode node) {
  // TODO: Recursive stacking is expensive.
  if (node.is_leaf()) {
    return node.payload();
  }
  if (node.degree() == 0) {
    return at::empty({0});
  }
  std::vector<at::Tensor> flat;
  for (auto child : node.unbind()) {
    flat.push_back(_to_tensor(child));
  }
  return stack(flat);
}

at::Tensor NestedTensor::to_tensor() {
  // TODO: Not necessarily a view because of stack and reshape.
  std::vector<int64_t> new_size;
  for (const auto& si : size()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t at
      // this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  if (is_contiguous()) {
    return (*_buffer).reshape(at::IntArrayRef(new_size));
  }
  return _to_tensor(_structure);
}

NestedTensor::NestedTensor(TensorNode&& structure)
    : _structure(structure),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(infer_nested_size(_structure)) {}

NestedTensor::NestedTensor(at::Tensor&& buffer, SizeNode nested_size)
    : _buffer(buffer),
      _structure(build_structure(*_buffer, nested_size)),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(nested_size) {}

} // namespace nested_tensor
} // namespace torch
