#include <functions.h>
#include <utils/nested_node_functions.h>

namespace torch {
namespace nested_tensor {

NestedTensor copy_(NestedTensor self, NestedTensor source, 
                bool non_blocking) {
  TORCH_CHECK(shape_matches(self, source), "self and source don't match in shape");

}

NestedTensor squeeze(NestedTensor input, c10::optional<int64_t> dim,
        c10::optional<NestedTensor> out) {
  auto sizes = input.sizes();
  TORCH_CHECK((sizes[dim]) && (*(sizes[dim]) == 1),
                  "Given dimension is either undefined or not a singleton.");
  if (out) {
    apply([dim](at::Tensor tensor) { return tensor.squeeze(*dim); },
                          input.get_structure());
    return out;
  }
  TensorNode result = map([dim](at::Tensor tensor) { return tensor.squeeze(*dim); },
                          input.get_structure()); 
  auto buffer = input.get_buffer();
  if (buffer) {
    return NestedTensor(std::move(result), std::move(*buffer));
  }
  return NestedTensor(std::move(result));
}

}
}
