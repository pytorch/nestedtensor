#include <nestedtensor/csrc/masking.h>
#include <chrono>
#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/padding.h>
#include <nestedtensor/csrc/cuda/attention.h>
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

std::vector<int64_t> get_max_size_from_efficient_size(EfficientSizeNode esize) {
  auto nt_opt_sizes = esize.opt_sizes();
  if (nt_opt_sizes.size() > 0 && *nt_opt_sizes[0] > 0) {
    auto sizes = esize.sizes();
    int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
    int64_t sizes_size_0 = sizes.size(0);
    int64_t sizes_size_1 = sizes.size(1);
    std::vector<int64_t> results(sizes_size_1, 0);
    TORCH_CHECK(sizes_size_1 > 0, "Internal error: Expected sizes_size_1 to be greater than 0.");
    for (int64_t i = 0; i < sizes_size_0; i++) {
      for (int64_t j = 0; j < sizes_size_1; j++) {
        int64_t val = sizes_ptr[i * sizes_size_1 + j];
        if (results[j] < val) {
          results[j] = val;
        }
      }
    }
    return results;
  }
  return _get_max_size(esize.to_size_node());
}

std::vector<int64_t> get_max_size(const Tensor& nt) {
  return get_max_size_from_efficient_size(get_efficient_nested_size(nt));
}


Tensor batch_offsets_from_efficient_size(EfficientSizeNode ef) {
  Tensor ef_sizes = ef.sizes();
  int64_t* nt_sizes_ptr = ef_sizes.data_ptr<int64_t>();
  Tensor offsets = torch::empty({1 + ef_sizes.size(0)}, torch::kInt64);
  int64_t* offsets_ptr = offsets.data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  int64_t ef_sizes_size_1 = ef_sizes.size(1);
  for (int64_t i = 0; i < ef_sizes.size(0); i++) {
    int64_t prod = 1;
    for (int64_t j = 0; j < ef_sizes_size_1; j++) {
      prod = prod * nt_sizes_ptr[i * ef_sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

std::vector<int64_t> padded_size_from_efficient_size(EfficientSizeNode ef_size) {
  Tensor nt_sizes = ef_size.sizes();
  auto max_size = get_max_size_from_efficient_size(ef_size);
  std::vector<int64_t> new_size;
  new_size.push_back(nt_sizes.size(0));
  for (int64_t i = 0; i < max_size.size(); i++) {
    new_size.push_back(max_size[i]);
  }
  return new_size;
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
          at::cumsum(nt_sizes, 0).to(torch::kInt32).reshape({-1});
      nt_sizes = at::cat({torch::tensor({0}, torch::kInt32), nt_sizes});
      Tensor output = torch::zeros(
          {*nt_opt_size[0], max_size_1, *nt_opt_size[2]}, nt_buffer.options());
      nt_sizes = nt_sizes.to(torch::kCUDA);
      Tensor output_mask = torch::zeros(
          {*nt_opt_size[0], max_size_1}, nt_buffer.options());
      output_mask = output_mask.to(torch::kInt32);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      if (nt.dtype() == torch::kFloat16) {
        nt_buffer = nt_buffer.to(torch::kFloat);
        output = output.to(torch::kFloat);
      }
      if (nt_buffer.dtype() == torch::kFloat) {
        nested_tensor::cuda::add_padding_mask_kernelLauncher<float>(
            nt_buffer.data_ptr<float>(),
            output.data_ptr<float>(),
            output_mask.data_ptr<int>(),
            nt_sizes.data_ptr<int>(),
            *nt_opt_size[0],
            output_mask.stride(0),
            output.stride(0),
            *nt_opt_size[2],
            defaultStream);
      }
      if (nt.dtype() == torch::kFloat16) {
        output = output.to(torch::kFloat16);
      }
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
    if (*mask_dim == 2 && get_dim(nt) == 3) {
      auto nt_size = get_efficient_nested_size(nt);
      auto esizes = nt_size.sizes();
      auto options = torch::TensorOptions().dtype(torch::kByte);
      auto result = torch::zeros({*opt_sizes[0], tmp_max_size[0]},
                                options);
      uint8_t* result_data = result.data_ptr<uint8_t>();
      int64_t* esizes_ptr = esizes.data_ptr<int64_t>();
      for (int64_t i = 0; i < esizes.size(0); i++) {
        int64_t length = esizes_ptr[i * esizes.size(1)];
        for (int64_t j = 0; j < length; j++) {
          result_data[i * result.size(1) + j] = 1;
        }
      }
      return result;
    }
    return _create_nt_mask(get_efficient_nested_size(nt), max_size);
  }
  max_size = get_max_size(nt);
  at::Tensor res_mask = _create_nt_mask(get_efficient_nested_size(nt), max_size);
  return merge_mask(res_mask, mask_dim);
}

Tensor from_padded_tensor(Tensor padded, EfficientSizeNode target_size) {
  TORCH_CHECK(padded.dim() == target_size.dim(),
      "Target size has different dimension as input padded Tensor.");
#ifdef WITH_CUDA
  if (padded.dim() > 1 && padded.dim() < 5 &&
      get_is_contiguous(padded) && padded.is_cuda()) {
    Tensor target_offsets = batch_offsets_from_efficient_size(target_size);
    std::vector<int64_t> padded_sizes = padded.sizes().vec();
    Tensor padded_sizes_tensor = torch::tensor(padded_sizes);
    Tensor output = torch::empty({target_size.numel()}, padded.options());
    Tensor target_size_sizes = target_size.sizes();

    at::Tensor metadata = at::cat({target_size_sizes.reshape(-1), padded_sizes_tensor, target_offsets});
    metadata = metadata.to(at::Device(kCUDA), torch::kInt32, true, true);

    std::vector<int64_t> split_sizes;
    split_sizes.push_back(target_size_sizes.numel());
    split_sizes.push_back(padded_sizes_tensor.numel());
    split_sizes.push_back(target_offsets.numel());

    std::vector<Tensor> split = at::split_with_sizes(metadata, IntArrayRef(split_sizes), 0);

    target_size_sizes = split[0];
    padded_sizes_tensor = split[1];
    target_offsets = split[2];

    at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
    if (padded.dtype() == torch::kFloat16) {
      nested_tensor::cuda::remove_padding_kernelLauncher(
          padded.data_ptr<c10::Half>(),
          output.data_ptr<c10::Half>(),
          target_offsets.data_ptr<int>(),
          padded_sizes_tensor.data_ptr<int>(),
          target_size_sizes.data_ptr<int>(),
          padded.dim() - 1,
          padded.size(0),
          defaultStream);
    }
    if (padded.dtype() == torch::kFloat) {
      nested_tensor::cuda::remove_padding_kernelLauncher(
          padded.data_ptr<float>(),
          output.data_ptr<float>(),
          target_offsets.data_ptr<int>(),
          padded_sizes_tensor.data_ptr<int>(),
          target_size_sizes.data_ptr<int>(),
          padded.dim() - 1,
          padded.size(0),
          defaultStream);
    }
    return wrap_buffer(std::move(output), target_size);
  }
#endif
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

Tensor _collapse_two_dims_3(Tensor input, int64_t dim1, int64_t dim2) {
  TORCH_CHECK(dim1 > 0, "dim1: Cannot collapse dim 0.");
  TORCH_CHECK(dim2 > 0, "dim2: Cannot collapse dim 0.");
  TORCH_CHECK(dim2 - 1 == dim1, "dim2 must be one more than dim1.")
  TORCH_CHECK(dim1 == 1, "dim1 must be 1.")
  TORCH_CHECK(get_dim(input) == 3, "Expected input to be 3 dim.");
  auto input_esizes = get_efficient_nested_size(input);
  Tensor nt_sizes = input_esizes.sizes();

  Tensor sizes_dim1 = at::native::narrow(nt_sizes, 1, 0, 1);
  Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 1, 1);

  Tensor new_nt_sizes;
  if (dim1 == 1) {
    Tensor collapsed_sizes = sizes_dim1 * sizes_dim2;
    new_nt_sizes = collapsed_sizes.contiguous();
  }
  auto new_esizes = torch::nested_tensor::EfficientSizeNode(input_esizes.structure(), new_nt_sizes);
  Tensor result = wrap_buffer(get_buffer(input), new_esizes);
  TORCH_CHECK(get_dim(result) == 2, "Expected result to be 2 dimensional.");
  return result;
}

Tensor to_padded_tensor(Tensor nt, double padding) {
#ifdef WITH_CUDA
  if ((get_dim(nt) >= 2 && get_dim(nt) <= 4)) {
    nt = NestedTensor_contiguous(nt, c10::MemoryFormat::Contiguous);
    auto nt_opt_size = get_opt_sizes(nt);
    auto orig_nt_dim = get_dim(nt);
    Tensor nt_buffer = get_buffer(nt);
    if (nt_buffer.is_cuda()) {
      if (get_dim(nt) == 3 && nt_opt_size[2]) {
        nt = _collapse_two_dims_3(nt, 1, 2);
      }
      auto esize = get_efficient_nested_size(nt);
      at::Tensor nt_sizes = esize.sizes();
      Tensor offsets = batch_offsets_from_efficient_size(esize);
      std::vector<int64_t> new_size = padded_size_from_efficient_size(esize);
      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
      Tensor output = at::empty(IntArrayRef(new_size), nt_buffer.options());

      int64_t input_dim = nt_sizes.size(1);
      int64_t batch_size = nt_sizes.size(0);
      at::Tensor metadata = at::cat({offsets, nt_sizes.reshape(-1)});
      metadata = metadata.to(at::Device(kCUDA), torch::kInt32, true, true);

      std::vector<int64_t> split_sizes;
      split_sizes.push_back(offsets.numel());
      split_sizes.push_back(nt_sizes.numel());

      std::vector<Tensor> split = at::split_with_sizes(metadata, IntArrayRef(split_sizes), 0);

      offsets = split[0];
      nt_sizes = split[1];

      if (nt_buffer.dtype() == torch::kFloat16) {
        nested_tensor::cuda::add_padding_kernelLauncher(
            nt_buffer.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            (c10::Half)(padding),
            offsets.data_ptr<int>(),
            nt_sizes.data_ptr<int>(),
            input_dim,
            new_size,
            batch_size,
            defaultStream);
        if (orig_nt_dim == 3 && nt_opt_size[2]) {
          output = output.reshape({output.size(0), -1, *nt_opt_size[2]});
        }
        return output;
      }
      if (nt_buffer.dtype() == torch::kFloat) {
        nested_tensor::cuda::add_padding_kernelLauncher(
            nt_buffer.data_ptr<float>(),
            output.data_ptr<float>(),
            (float)(padding),
            offsets.data_ptr<int>(),
            nt_sizes.data_ptr<int>(),
            input_dim,
            new_size,
            batch_size,
            defaultStream);
        if (orig_nt_dim == 3 && nt_opt_size[2]) {
          output = output.reshape({output.size(0), -1, *nt_opt_size[2]});
        }
        return output;
      }
      return output;
      TORCH_CHECK(false, "Input datatype ", nt_buffer.dtype(), " is not supported.");
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
