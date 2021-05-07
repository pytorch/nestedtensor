#pragma once
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<std::vector<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

struct PackedStorage {
  explicit PackedStorage(TensorNode structure) :
    _structure (structure){
    }
private:
  TensorNode _structure;
  at::Tensor _first_variable;
  SizeNode _nested_size;
  std::vector<int64_t> _sizes;
};

}
}
