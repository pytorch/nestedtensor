
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

at::Tensor to_tensor(NestedTensorImpl* nt_impl) {
  // TODO: Not necessarily a view because of stack and reshape.
  std::vector<int64_t> new_size;
  for (const auto& si : nt_impl->opt_sizes()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t at
      // this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  return _to_tensor(nt_impl->get_structure());
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_) {
  if (!dim_) {
    return NestedTensor_to_tensor(tensor, 0);
  }
  int64_t dim = maybe_wrap_dim((*dim_), get_dim(tensor));
  if (dim != 0) {
    TORCH_CHECK(false, "Non-zero dimension ", *dim_, " is currently not supported.");
  }
  std::vector<int64_t> new_size;
  auto impl_data = get_nested_tensor_impl(tensor);
  for (const auto& si : impl_data->opt_sizes()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t
      // at this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  return _to_tensor(impl_data->get_structure());
  // // If dim is bigger than nested_dim the NestedTensor is already
  // // of Tensor for dimensions bigger than the given.
  // if (impl_data->nested_dim() == 1) {
  //   return tensor;
  // }
  // // At this point nested_dim is at least 2. That means any unbind
  // // operation of a child must yield NestedTensors.
  // // If dim is 1 then we'll apply to_tensor(0) to the children and must expect
  // // Tensors.
  // std::vector<at::Tensor> unbound = at::unbind(tensor, 0);
  // std::vector<TensorNode> result;
  // for (Tensor child : unbound) {
  //   auto ci = NestedTensor_to_tensor(child, dim - 1);
  //   if (is_nested_tensor_impl(ci)) {
  //     auto s = get_nested_tensor_impl(ci)->get_structure();
  //     result.push_back(TensorNode(std::move(s)));
  //   } else {
  //     // TODO: If it's a NestedTensor instance get the structure
  //     result.push_back(TensorNode(std::move(ci)));
  //   }
  // }
  // return wrap_tensor_node(TensorNode(std::move(result)));
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def("to_tensor(Tensor tensor, int? dim) -> Tensor");
  m.impl("to_tensor", NestedTensorKey,
  [](Tensor tensor, c10::optional<int64_t> dim) {
    return NestedTensor_to_tensor(tensor, dim);
  });
  m.impl("to_tensor", c10::DispatchKey::CPU,
  [](Tensor tensor, c10::optional<int64_t> dim) {
    return NestedTensor_to_tensor(tensor, dim);
  });
}

} // namespace at
