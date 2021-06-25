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
  if ((is_last_size_sum + is_zero_sum) == get_numel(collapsed_mask)) {
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

Tensor pad_tensor_to_shape(Tensor t, const std::vector<int64_t>& goal_shape, double value = 0) {
  std::vector<int64_t> padd;
  auto tup = t.sizes();
  if (get_dim(t) != (int64_t)(goal_shape.size())) {
    throw std::runtime_error("dimension doesn't match length of goal shape.");
  }
  for (int64_t i = tup.size() - 1; i >= 0; i--) {
    padd.push_back(0);
    padd.push_back(goal_shape[i] - tup[i]);
  }
  Tensor new_tensor = at::constant_pad_nd(t, IntArrayRef(padd), value);
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

std::vector<int64_t> get_max_size(const Tensor& nt) {
  if (get_nested_dim(nt) == 1) {
    auto nt_opt_sizes = get_opt_sizes(nt);
    if (nt_opt_sizes.size() > 0 && *nt_opt_sizes[0] > 0) {
      auto esize = get_efficient_nested_size(nt);
      auto sizes = esize.sizes();
      auto max_sizes = std::get<0>(sizes.max(0));
      std::vector<int64_t> result;
      for (int64_t i = 0; i < max_sizes.size(0); i++) {
        result.push_back(max_sizes[i].item<int64_t>());
      }
      return result;
    }
  }
  return _get_max_size(get_nested_size(nt));
}

std::tuple<Tensor, Tensor> pad_nt(Tensor nt, std::vector<int64_t> shape) {
  if (!is_nested_tensor_impl(nt)) {
    if (get_numel(nt) == 0) {
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
    if ((get_numel(mask) == 0) || (get_numel(mask) == 1 && mask.item<bool>())) {
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
        if (get_numel(tmp) > 0) {
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
  TORCH_CHECK(nested_dim == 1, "Only nested_dim of 1 is currently supported.");
  std::vector<c10::optional<Tensor>> inner_tensors;
  if ((get_numel(mask) == 0) || (get_numel(mask) == 1 && mask.item<bool>())) {
    for (int64_t i = 0; i < tensor.size(0); i++) {
      inner_tensors.push_back(
          nt_from_tensor_mask(tensor[i], mask, nested_dim - 1));
    }
  } else if (get_numel(mask) == 1 && !mask.item<bool>()) {
    inner_tensors.push_back(c10::nullopt);
  } else {
    for (int64_t i = 0; i < tensor.size(0); i++) {
      inner_tensors.push_back(
          nt_from_tensor_mask(tensor[i], mask[i], nested_dim - 1));
    }
  }
  std::vector<TensorNode> inner_tensor_nodes;
  for (size_t i = 0; i < inner_tensors.size(); i++) {
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
#ifdef WITH_CUDA
  if (get_dim(nt) == 3 && get_is_contiguous(nt) && mask_dim && *mask_dim == 2) {
    auto nt_opt_size = get_opt_sizes(nt);
    Tensor nt_buffer = get_buffer(nt);
    if (nt_opt_size[2] && nt_buffer.is_cuda()) {
      Tensor nt_sizes_ =
          get_efficient_nested_size(nt).sizes().to(torch::kInt32);
      TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor metadata of unexpected dimension.")
      Tensor nt_sizes = at::native::narrow(nt_sizes_, 1, 0, 1);
      int max_size_1 = nt_sizes.max().item<int>();
      nt_sizes =
          at::native::cumsum(nt_sizes, 0).to(torch::kInt32).reshape({-1});
      nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes});
      Tensor output = torch::zeros(
          {*nt_opt_size[0], max_size_1, *nt_opt_size[2]}, nt_buffer.options());
      nt_sizes = nt_sizes.to(torch::kCUDA);
      Tensor output_mask = torch::zeros(
          {*nt_opt_size[0], max_size_1}, nt_buffer.options());
      output_mask = output_mask.to(torch::kInt32);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      nested_tensor::cuda::add_padding_mask_kernelLauncher(
          nt_buffer.data_ptr<float>(),
          output.data_ptr<float>(),
          output_mask.data_ptr<int>(),
          nt_sizes.data_ptr<int>(),
          *nt_opt_size[0],
          output_mask.stride(0),
          output.stride(0),
          *nt_opt_size[2],
          defaultStream);
      return std::make_tuple(output, output_mask.to(torch::kBool));
    }
  }
#endif
  TORCH_CHECK(
      !mask_dim || *mask_dim <= get_dim(nt),
      "Requested mask dimension ",
      *mask_dim,
      " is bigger than dimension ",
      get_dim(nt),
      " of given NestedTensor.");

  auto opt_sizes = get_opt_sizes(nt);
  if (opt_sizes.size() == 1 && *opt_sizes[0] == 1) {
    nt = NestedTensor_contiguous(nt);
    Tensor nt_buffer = get_buffer(nt);
    nt_buffer = nt_buffer.reshape({-1});
    Tensor result_mask = !mask_dim || *mask_dim == 0 ? torch::tensor(true)
                                                     : torch::tensor({true});
    return std::make_tuple(nt_buffer, result_mask);
  }

  auto max_size = get_max_size(nt);
  at::Tensor res_tensor;
  at::Tensor res_mask;
  std::tie(res_tensor, res_mask) = pad_nt(nt, max_size);
  return merge_tensor_mask(res_tensor, res_mask, mask_dim);
}

Tensor merge_mask(
    Tensor mask,
    c10::optional<int64_t> mask_dim) {
  if (mask_dim && get_dim(mask) == (*mask_dim)) {
    return  mask;
  }

  if (get_dim(mask) == 0) {
    return mask;
  }

  int64_t last_size = mask.size(-1);
  Tensor collapsed_mask = mask.sum(-1);
  Tensor is_last_size = (collapsed_mask == last_size);
  Tensor is_zero = (collapsed_mask == 0);
  int64_t is_last_size_sum = is_last_size.sum().item<int64_t>();
  int64_t is_zero_sum = is_zero.sum().item<int64_t>();
  if ((is_last_size_sum + is_zero_sum) == get_numel(collapsed_mask)) {
    collapsed_mask = collapsed_mask.to(torch::kBool);
    return merge_mask(collapsed_mask, mask_dim);
  }

  if (mask_dim && mask_dim != get_dim(mask)) {
    throw std::runtime_error(
        "Mask dimension is too small to represent data tensor.");
  }
  // This is expected to be a no-op, except in rare cases.
  mask = mask.contiguous();
  return mask;
}

Tensor _create_nt_mask(std::vector<int64_t> sizes, std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    numel = numel * sizes[i];
  }
  TORCH_CHECK(numel > 0, "Empty tensors are not yet supported.");
  // Dont pad in case of a scalar
  if (sizes.size() == 0) {
    return torch::tensor(true);
  }
  auto options = torch::TensorOptions().dtype(torch::kByte);
  Tensor mask = pad_tensor_to_shape(
      torch::full(
          IntArrayRef(sizes),
          true,
          options),
      shape);
  return mask;
}

Tensor _create_nt_mask(SizeNode nt_size, std::vector<int64_t> shape) {
  if (nt_size.degree() == 0) {
    return _create_nt_mask(nt_size.payload(), shape);
  }

  std::vector<Tensor> res_mask;
  if (nt_size.degree() == 0) {
    return torch::tensor({false}, torch::kByte);
  } else {
    for (auto child : nt_size.unbind()) {
      Tensor mask = _create_nt_mask(child, shape);
      res_mask.push_back(mask);
    }
  }

  return at::stack(res_mask);
}

Tensor _create_nt_mask(EfficientSizeNode nt_size, std::vector<int64_t> shape) {
  if (nt_size.height() == 1) {
    std::vector<at::Tensor> tmp_masks;
    auto esizes = nt_size.sizes();
    int64_t* esizes_ptr = esizes.data_ptr<int64_t>();
    for(int64_t i = 0; i < esizes.size(0); i++) {
      std::vector<int64_t> tmp_sizes;
      for(size_t j = 0; j < shape.size(); j++) {
        tmp_sizes.push_back(esizes_ptr[i * esizes.stride(0) + j]);
      }
      tmp_masks.push_back(_create_nt_mask(tmp_sizes, shape));
    }
    return at::stack(tmp_masks);
  }
  return _create_nt_mask(nt_size.to_size_node(), shape);
}

Tensor to_mask(
    Tensor nt,
    c10::optional<int64_t> mask_dim) {
  TORCH_CHECK(
      !mask_dim || *mask_dim <= get_dim(nt),
      "Requested mask dimension ",
      *mask_dim,
      " is bigger than dimension ",
      get_dim(nt),
      " of given NestedTensor.");


  auto opt_sizes = get_opt_sizes(nt);
  if (opt_sizes.size() == 1 && *opt_sizes[0] == 1) {
    Tensor result_mask = !mask_dim || *mask_dim == 0 ? torch::tensor(true)
                                                     : torch::tensor({true});
    return result_mask;
  }

  std::vector<int64_t> max_size;
  if (get_nested_dim(nt) == 1 &&
      get_dim(nt) > 1 &&
      mask_dim &&
      *mask_dim > 1) {
    auto tmp_max_size = get_max_size(nt);
    for (int64_t i = 1; i < *mask_dim; i++) {
      max_size.push_back(tmp_max_size[i - 1]);
    }
    return _create_nt_mask(get_efficient_nested_size(nt), max_size);
  }
  max_size = get_max_size(nt);
  at::Tensor res_mask = _create_nt_mask(get_efficient_nested_size(nt), max_size);
  return merge_mask(res_mask, mask_dim);
}

Tensor from_padded_tensor(Tensor padded, EfficientSizeNode target_size,
    EfficientSizeNode target_stride) {
#ifdef WITH_CUDA
  if (padded.dim() == 3 && target_size.dim() == 3 && get_is_contiguous(padded)) {
    auto nt_opt_size = target_size.opt_sizes();
    if (nt_opt_size[2] && padded.is_cuda()) {
      Tensor nt_sizes_ = target_size.sizes().to(torch::kInt32);
      TORCH_CHECK(nt_sizes_.dim() == 2, "NestedTensor must be of nested_dim 2.")
      Tensor nt_sizes = at::native::narrow(nt_sizes_, 1, 0, 1);
      int max_size_1 = nt_sizes.max().item<int>();
      nt_sizes =
          at::native::cumsum(nt_sizes, 0).to(torch::kInt32).reshape({-1});
      nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes});
      Tensor output = torch::empty({target_size.numel()}, padded.options());
      nt_sizes = nt_sizes.to(torch::kCUDA);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      nested_tensor::cuda::remove_padding_kernelLauncher(
          padded.data_ptr<float>(),
          output.data_ptr<float>(),
          nt_sizes.data_ptr<int>(),
          *nt_opt_size[0],
          padded.stride(0),
          *nt_opt_size[2],
          defaultStream);
      return wrap_buffer(std::move(output), target_size, target_stride);
    }
  }
#endif
  TORCH_CHECK(false, "from_padded_tensor not implemented for this case.");
}

Tensor from_padded_tensor(Tensor padded, EfficientSizeNode target_size) {
  at::Tensor target_size_tensor = std::get<0>(at::max(target_size.sizes(), 0));
  std::vector<int64_t> target_size_vec(target_size_tensor.data_ptr<int64_t>(),
      target_size_tensor.data_ptr<int64_t>() + target_size_tensor.numel());
  std::vector<at::Tensor> masks;
  std::vector<at::Tensor> all_sizes = target_size.sizes().unbind();
  for (int64_t i = 0; i < all_sizes.size(); i++) {
    std::vector<int64_t> sizes_i(
        all_sizes[i].data_ptr<int64_t>(), 
        all_sizes[i].data_ptr<int64_t>() + all_sizes[i].numel());
    at::Tensor mask_i = padded.new_full(
                                    IntArrayRef(sizes_i),
                                    true,
                                    torch::kByte,
                                    c10::nullopt,
                                    c10::nullopt,
                                    c10::nullopt);
    mask_i = pad_tensor_to_shape(mask_i, target_size_vec);
    masks.push_back(mask_i);
  }
  at::Tensor final_mask = at::stack(masks);
  at::Tensor new_buffer = padded.masked_select(final_mask);
  return wrap_buffer(std::move(new_buffer), target_size);
}

Tensor to_padded_tensor(Tensor nt, double padding) {
#ifdef WITH_CUDA
  if (get_dim(nt) == 3 && get_is_contiguous(nt)) {
    auto nt_opt_size = get_opt_sizes(nt);
    Tensor nt_buffer = get_buffer(nt);
    if (nt_opt_size[2] && nt_buffer.is_cuda()) {
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
  auto opt_sizes = get_opt_sizes(nt);
  if (opt_sizes.size() == 1 && *opt_sizes[0] == 1) {
    nt = NestedTensor_contiguous(nt);
    return get_buffer(nt);
  }
  auto max_size = get_max_size(nt);
  TensorNode structure = get_nested_tensor_structure(nt);
  if (structure.degree() == 0) {
    return torch::tensor({padding});
  }
  std::vector<Tensor> res_tensor;
  for (auto child : structure.unbind()) {
    at::Tensor tensor = child.payload();
    if (get_numel(tensor) == 0) {
      TORCH_CHECK(false, "Empty tensors are not yet supported.");
    }
    // Dont pad in case of a scalar
    if (get_dim(tensor) == 0) {
      res_tensor.push_back(tensor);
    }
    res_tensor.push_back(pad_tensor_to_shape(tensor, max_size, padding));
  }
  return at::stack(res_tensor);
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

  m.def("to_tensor_mask(Tensor nt, int? mask_dim) -> (Tensor, Tensor)");
  m.impl("to_tensor_mask", NestedTensorKey, to_tensor_mask);

  m.def("to_mask(Tensor nt, int? mask_dim) -> Tensor");
  m.impl("to_mask", NestedTensorKey, to_mask);

  m.def("to_padded_tensor(Tensor nt, float padding) -> Tensor");
  m.impl("to_padded_tensor", NestedTensorKey, to_padded_tensor);
}
