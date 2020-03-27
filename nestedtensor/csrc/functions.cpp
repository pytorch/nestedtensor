#include <functions.h>
#include <utils/nested_node_functions.h>
#include <torch/extension.h>

namespace torch {
namespace nested_tensor {

TensorNode _squeeze_nested_dim(TensorNode structure,
    int64_t dim) {
  if (dim == 0) {
    return structure.children(0);
  }
  return TensorNode(_squeeze_nested_dim(structure, dim - 1));
}

// TODO: If size(0) is 1 and we squeeze should this turn into a Tensor?
// Squeeze doens't touch the underlying data and is effectively a meta-data
// operation
NestedTensor squeeze(
    NestedTensor input,
    c10::optional<int64_t> dim_,
    c10::optional<NestedTensor> out) {
  auto sizes = input.sizes();
  if (dim_) {
    int64_t dim = at::maybe_wrap_dim(*dim_, input.dim());
    std::cout << "Dim: " << dim << std::endl;
    TORCH_CHECK(
        ((sizes[dim]) && ((*(sizes[dim])) == 1)),
        "Given dimension is either undefined or not a singleton.");
    TensorNode structure = input.get_structure();
    int64_t nested_dim = input.nested_dim();
    if (dim < nested_dim) {
      structure = _squeeze_nested_dim(structure, dim);
    } else {
      int64_t height = structure.height();
      structure = map(
          [dim, height](at::Tensor tensor) {
            if (dim) {
              return tensor.squeeze(dim - height);
            }
            return tensor.squeeze();
          },
          structure);
    }
    if (input.get_buffer()) {
      at::Tensor buffer = *input.get_buffer();
      input = NestedTensor(std::move(buffer), std::move(structure));
    } else {
      input = NestedTensor(std::move(structure));
    }
  } else {
    // TODO: First dimension is always ignored.
    for (size_t i = 0; i < sizes.size() - 1; i++) {
      size_t index = sizes.size() - i - 1;
      c10::optional<int64_t> s = sizes[index];
      if (s) {
        std::cout << " s: " << *s;
      }
      if (s && ((*s) == 1)) {
        input = squeeze(input, index, c10::nullopt);
      }
    }
  }
  if (out) {
    (*out).copy_(input);
    return *out;
  }
  return input;
}
}
}
