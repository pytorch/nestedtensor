#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {
struct EfficientSizeNode {
 private:
  int64_t _height;
  int64_t _tensor_dim;
  int64_t num_leaf;
}
}
}
