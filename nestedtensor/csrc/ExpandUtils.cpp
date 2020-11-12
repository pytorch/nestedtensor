
#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;
using namespace c10;

bool NestedTensor_sizes_equal_nt_other(
    const Tensor& self,
    IntArrayRef nested_size_other) {
  auto tmp =
      torch::nested_tensor::deserialize_size_node(nested_size_other.vec(), 0);
  SizeNode nested_size = std::get<1>(tmp);
  if (is_nested_tensor_impl(self)) {
    // std::cout << "SE1" << std::endl;
    return false;
    // return torch::nested_tensor::shape_matches(
    //     get_nested_tensor_impl(self)->nested_size(), nested_size);
  }
  // std::cout << "SE2" << std::endl;
  return false;
}

bool NestedTensor_sizes_equal(const Tensor& self, IntArrayRef size_other) {
  TORCH_CHECK(false, "NOT SUPPORTED YET.");
  return self.sizes().equals(size_other);
}

// Can nested_size_other be expanded to match the shape of grad?
// If this is true, a call to sum_to_nt will follow next in autograd/engine.cpp
// to reduce grad down to the shape of nested_size_other.
bool NestedTensor_native_is_expandable_to_nt_other(
    const Tensor& grad,
    IntArrayRef nested_size_other) {
  auto tmp =
      torch::nested_tensor::deserialize_size_node(nested_size_other.vec(), 0);
  SizeNode nested_size = std::get<1>(tmp);
  if (is_nested_tensor_impl(grad)) {
    std::cout << "NTNE1" << std::endl;
    return false;
    // return torch::nested_tensor::shape_matches(
    //     get_nested_tensor_impl(self)->nested_size(), nested_size);
  }
  std::cout << "grad: " << grad << std::endl;
  auto asdf = map([](c10::List<int64_t> s) {
      for (size_t i = 0; i < s.size(); i++) {
        std::cout << "s[" << i << "]: " << s[i] << std::endl;
      }
      return s;
    }, nested_size);
  std::cout << "NTNE2" << std::endl;
  return false;
}

bool NestedTensor_native_is_expandable_to(
    const Tensor& grad,
    IntArrayRef metadata_shape) {
  std::cout << "2830283" << std::endl;
  return true;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "sizes_equal", NestedTensor_sizes_equal);
  nt_impl(m, "native_is_expandable_to", NestedTensor_native_is_expandable_to);
}
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  nt_impl(m, "sizes_equal_nt_other", NestedTensor_sizes_equal_nt_other);
  nt_impl(
      m,
      "native_is_expandable_to_nt_other",
      NestedTensor_native_is_expandable_to_nt_other);
}
} // namespace at
