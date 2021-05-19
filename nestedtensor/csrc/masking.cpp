#include <nestedtensor/csrc/masking.h>
#include <chrono>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/padding.h>
#endif

using namespace torch::nested_tensor;
using namespace at;

std::tuple<Tensor, Tensor> merge_tensor_mask(
    Tensor tensor,
    Tensor mask,
    c10::optional<int64_t> mask_dim) {
  if (mask_dim && get_dim(mask) == (*mask_dim)) {
    return std::make_tuple(tensor, mask);
  }

  if (get_dim(mask) == 0) {
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

  if (mask_dim && mask_dim != get_dim(mask)) {
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
  if (get_dim(t) != goal_shape.size()) {
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

std::vector<int64_t> _get_max_size(const SizeNode& size_node) {
  std::vector<int64_t> result;
  if (size_node.is_leaf()) {
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  if (size_node.degree() > 0) {
    std::vector<int64_t> first_size = _get_max_size(size_node.children(0));
    for (const auto& size : first_size) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      std::vector<int64_t> ith_size = _get_max_size(size_node.children(i));
      for (size_t j = 0; j < ith_size.size(); j++) {
        result[j] = result[j] > ith_size[j] ? result[j] : ith_size[j];
      }
    }
  }
  return result;
}

std::vector<int64_t> get_max_size(Tensor nt) {
  return _get_max_size(get_nested_size(nt));
}

Tensor NestedTensor_to_padded_tensor(Tensor nt, double padding) {
#ifdef WITH_CUDA
  if (get_dim(nt) == 3) {
    auto nt_opt_size = get_opt_sizes(nt);
    if (nt_opt_size[2]) {
      Tensor nt_buffer = get_buffer(nt);
      Tensor nt_sizes_ =
          get_efficient_nested_size(nt).sizes().to(torch::kInt32);
      TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor must be of nested_dim 2.")
      Tensor nt_sizes = at::native::narrow(nt_sizes_, 1, 0, 1);
      int max_size_1 = nt_sizes.max().item<int>();
      nt_sizes =
          at::native::cumsum(nt_sizes, 0).to(torch::kInt32).reshape({-1});
      nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes});
      Tensor output = torch::empty(
          {*nt_opt_size[0], max_size_1, *nt_opt_size[2]}, nt_buffer.options());
      output.fill_(padding);
      nt_sizes = nt_sizes.to(torch::kCUDA);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      nested_tensor::cuda::add_padding_kernelLauncher(
          nt_buffer.data_ptr<float>(),
          output.data_ptr<float>(),
          nt_sizes.data_ptr<int>(),
          *nt_opt_size[0],
          output.stride(0),
          *nt_opt_size[2],
          defaultStream);
      return output;
    }
  }
#endif
}

std::tuple<Tensor, Tensor> pad_nt(Tensor nt, std::vector<int64_t> shape) {
  if (!is_nested_tensor_impl(nt)) {
    if (nt.numel() == 0) {
      TORCH_CHECK(false, "Empty tensors are not yet supported.");
    }
    // Dont pad in case of a scalar
    if (get_dim(nt) == 0) {
      return std::make_tuple(nt, torch::tensor(true));
    }

    Tensor tensor = pad_tensor_to_shape(nt, shape);
    Tensor mask = pad_tensor_to_shape(
        nt.new_full(
            nt.sizes(),
            true,
            torch::kByte,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt),
        shape);
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
      std::tie(tensor, mask) =
          pad_nt(wrap_tensor_node(std::move(child)), shape);
      res_tensor.push_back(tensor);
      res_mask.push_back(mask);
    }
  }

  return std::make_tuple(at::stack(res_tensor), at::stack(res_mask));
}

c10::optional<Tensor> nt_from_tensor_mask(
    Tensor tensor,
    Tensor mask,
    int64_t nested_dim) {
  if (nested_dim == 0) {
    if ((mask.numel() == 0) || (mask.numel() == 1 && mask.item<bool>())) {
      return tensor;
    }

    if (get_dim(mask) == 1) {
      std::vector<Tensor> tensors;
      for (int64_t i = 0; i < mask.size(0); i++) {
        if (mask[i].item<bool>()) {
          tensors.push_back(tensor[i]);
        }
      }
      if (tensors.size() == 0) {
        return torch::tensor({}).to(tensor);
      }
      return at::stack(tensors);
    }

    if (get_dim(mask) > 1) {
      std::vector<Tensor> tensors;
      bool all_zero = true;
      for (int64_t i = 0; i < mask.size(0); i++) {
        Tensor tmp = *nt_from_tensor_mask(tensor[i], mask[i], nested_dim);
        if (tmp.numel() > 0) {
          all_zero = false;
          tensors.push_back(tmp);
        }
      }
      if (all_zero) {
        for (int64_t i = 0; i < mask.size(0); i++) {
          Tensor tmp = *nt_from_tensor_mask(tensor[i], mask[i], nested_dim);
          tensors.push_back(tmp);
        }
      }
      if (tensors.size() == 0) {
        return torch::tensor({}).to(tensor);
      }
      return at::stack(tensors);
    }
    return c10::nullopt;
  }
  std::vector<c10::optional<Tensor>> inner_tensors;
  if ((mask.numel() == 0) || (mask.numel() == 1 && mask.item<bool>())) {
    for (int64_t i = 0; i < tensor.size(0); i++) {
      inner_tensors.push_back(
          nt_from_tensor_mask(tensor[i], mask, nested_dim - 1));
    }
  } else if (mask.numel() == 1 && !mask.item<bool>()) {
    inner_tensors.push_back(c10::nullopt);
  } else {
    for (int64_t i = 0; i < tensor.size(0); i++) {
      inner_tensors.push_back(
          nt_from_tensor_mask(tensor[i], mask[i], nested_dim - 1));
    }
  }
  std::vector<TensorNode> inner_tensor_nodes;
  for (int64_t i = 0; i < inner_tensors.size(); i++) {
    if (inner_tensors[i]) {
      TensorNode node = get_nested_tensor_structure(*inner_tensors[i]);
      inner_tensor_nodes.push_back(node);
    }
  }
  return wrap_tensor_node(TensorNode(std::move(inner_tensor_nodes)));
}

std::tuple<Tensor, Tensor> to_tensor_mask(
    Tensor nt,
    c10::optional<int64_t> mask_dim) {
  // TODO: Cover if not isinstance(nt, list) and nt.size() == (1,):
  // TODO: Move to_tensor_mask entirely into C++

  std::vector<int64_t> max_size = get_max_size(nt);
  Tensor tensor;
  Tensor mask;
  std::tie(tensor, mask) = pad_nt(nt, max_size);
  std::tie(tensor, mask) = merge_tensor_mask(tensor, mask, mask_dim);
  return std::make_tuple(tensor, mask);
}

TORCH_LIBRARY_FRAGMENT(nestedtensor, m) {
  m.def(
      "merge_tensor_mask(Tensor tensor, Tensor mask, int? mask_dim=None) -> (Tensor, Tensor)");
  m.impl("merge_tensor_mask", TORCH_FN(merge_tensor_mask));

  m.def("pad_nt(Tensor nt, int[] shape) -> (Tensor, Tensor)");
  m.impl("pad_nt", NestedTensorKey, TORCH_FN(pad_nt));

  m.def(
      "nt_from_tensor_mask(Tensor tensor, Tensor mask, int nested_dim) -> Tensor?");
  m.impl("nt_from_tensor_mask", TORCH_FN(nt_from_tensor_mask));

  m.def("get_max_size(Tensor nt) -> int[]");
  m.impl("get_max_size", NestedTensorKey, TORCH_FN(get_max_size));

  m.def("to_padded_tensor(Tensor nt, float padding) -> Tensor");
  m.impl("to_padded_tensor", NestedTensorKey, NestedTensor_to_padded_tensor);
}
