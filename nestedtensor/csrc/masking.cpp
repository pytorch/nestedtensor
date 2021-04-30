#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_functions.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>
#include <chrono>

using namespace torch::nested_tensor;
using namespace at;

std::tuple<Tensor, Tensor> merge_tensor_mask(
    Tensor tensor,
    Tensor mask,
    c10::optional<int64_t> mask_dim) {
  if (mask_dim && mask.dim() == (*mask_dim)) {
    return std::make_tuple(tensor, mask);
  }

  if (mask.dim() == 0) {
    return std::make_tuple(tensor, mask);
  }

  int64_t last_size = mask.size(-1);
  Tensor collapsed_mask = mask.sum(-1);
  Tensor is_last_size = (collapsed_mask == last_size);
  Tensor is_zero = (collapsed_mask == 0);
  int64_t is_last_size_sum = is_last_size.sum().item<int64_t>();
  int64_t is_zero_sum = is_zero.sum().item<int64_t>();
  if ((is_last_size_sum + is_zero_sum) == collapsed_mask.numel()) {
    collapsed_mask = collapsed_mask.to(torch::kBool);
    return merge_tensor_mask(tensor, collapsed_mask, mask_dim);
  }

  if (mask_dim && mask_dim != mask.dim()) {
    throw std::runtime_error(
        "Mask dimension is too small to represent data tensor.");
  }
  // This is expected to be a no-op, except in rare cases.
  tensor = tensor.contiguous();
  mask = mask.contiguous();
  return std::make_tuple(tensor, mask);
}

Tensor pad_tensor_to_shape(Tensor t, std::vector<int64_t> goal_shape) {
  std::vector<int64_t> padd;
  auto tup = t.sizes();
  if (t.dim() != goal_shape.size()) {
    throw std::runtime_error("dimension doesn't match length of goal shape.");
  }
  for (int64_t i = tup.size() - 1; i >= 0; i--) {
    padd.push_back(0);
    padd.push_back(goal_shape[i] - tup[i]);
  }
  Tensor new_tensor = at::constant_pad_nd(t, IntArrayRef(padd), 0);
  new_tensor = new_tensor.reshape(IntArrayRef(goal_shape));
  return new_tensor;
}

std::tuple<Tensor, Tensor> pad_nt(Tensor nt, std::vector<int64_t> shape) {
  if (!is_nested_tensor_impl(nt)) {
    if (nt.numel() == 0) {
      throw std::runtime_error("Empty tensors are not yet supported.");
    }
    // Dont pad in case of a scalar
    if (nt.dim() == 0) {
      return std::make_tuple(nt, torch::tensor(true));
    }

    Tensor tensor = pad_tensor_to_shape(nt, shape);
    Tensor mask = pad_tensor_to_shape(
        nt.new_full(nt.sizes(), true, torch::kByte, c10::nullopt,
          c10::nullopt, c10::nullopt), shape);
    return std::make_tuple(tensor, mask);
  }

  std::vector<Tensor> res_tensor;
  std::vector<Tensor> res_mask;
  TensorNode structure = get_nested_tensor_structure(nt);
  if (structure.degree() == 0) {
    return std::make_tuple(
        torch::tensor({0}), torch::tensor({false}, torch::kByte));
  } else {
    for (auto child : structure.unbind()) {
      Tensor tensor;
      Tensor mask;
      std::tie(tensor, mask) = pad_nt(wrap_tensor_node(std::move(child)), shape);
      res_tensor.push_back(tensor);
      res_mask.push_back(mask);
    }
  }

  return std::make_tuple(at::stack(res_tensor), at::stack(res_mask));
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def(
      "merge_tensor_mask(Tensor tensor, Tensor mask, int? mask_dim=None) -> (Tensor, Tensor)");
  m.impl("merge_tensor_mask", TORCH_FN(merge_tensor_mask));

  m.def(
      "pad_nt(Tensor nt, int[] shape) -> (Tensor, Tensor)");
  m.impl("pad_nt", NestedTensorKey, TORCH_FN(pad_nt));
}
