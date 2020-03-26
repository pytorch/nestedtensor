#include <functions.h>
#include <utils/nested_node_functions.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

// TODO: Make dim work for nested dimensions.
NestedTensor squeeze(
    NestedTensor input,
    c10::optional<int64_t> dim,
    c10::optional<NestedTensor> out) {
  auto sizes = input.sizes();
  if (dim) {
    TORCH_CHECK(
        ((sizes[*dim]) && ((*(sizes[*dim])) == 1)),
        "Given dimension is either undefined or not a singleton.");
  }
  int64_t nested_dim = input.nested_dim();
  TensorNode result = map(
      [dim](at::Tensor tensor) {
        if (dim) {
          return tensor.squeeze(*dim);
        }
        return tensor.squeeze();
      },
      input.get_structure());
  if (out) {
    (*out).copy_(NestedTensor(std::move(result)));
    return *out;
  }
  // Squeeze doens't touch the underlying data and is effectively a meta-data
  // operation
  // so we can copy the buffer as is.
  auto buffer = input.get_buffer();
  if (buffer) {
    return NestedTensor(std::move(*buffer), std::move(result));
  }
  return NestedTensor(std::move(result));
}
}
}
