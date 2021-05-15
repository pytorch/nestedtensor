#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  auto nt_impl = get_nested_tensor_impl(self);
  int64_t nested_dim = nt_impl->nested_dim();
  dim = maybe_wrap_dim(dim, get_dim(self));
  TORCH_CHECK(
      dim >= nested_dim, "cumsum of nested dimensions is not implemented yet.");
  return map_nested_tensor(
      [nested_dim, dim](at::Tensor tensor) {
        return at::cumsum(tensor, dim - nested_dim);
      },
      self);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> make_split_dims(
    const Tensor& self,
    c10::ArrayRef<int64_t> dims) {
  auto nt_impl = get_nested_tensor_impl(self);
  int64_t nested_dim = nt_impl->nested_dim();
  std::vector<int64_t> tensordims;
  std::vector<int64_t> nesteddims;
  for (size_t i = 0; i < dims.size(); i++) {
    int64_t dim = maybe_wrap_dim(dims[i], get_dim(self));
    if (dim < nested_dim) {
      nesteddims.push_back(dim);
    } else {
      tensordims.push_back(dim - nested_dim);
    }
  }
  std::sort(tensordims.begin(), tensordims.end());
  std::sort(nesteddims.begin(), nesteddims.end());
  return std::make_tuple(tensordims, nesteddims);
}

template <typename F>
Tensor NestedTensor_func_dim(
    F& fn,
    const Tensor& self,
    c10::ArrayRef<int64_t> dims,
    bool keepdims,
    c10::optional<ScalarType> dtype) {
  std::vector<int64_t> tensordims;
  std::vector<int64_t> nesteddims;
  std::tie(tensordims, nesteddims) = make_split_dims(self, dims);
  at::Tensor output = self;
  if (tensordims.size() > 0) {
    output = map_nested_tensor(
        [fn, tensordims, keepdims, dtype](at::Tensor tensor) {
          return fn(
              tensor, c10::ArrayRef<int64_t>(tensordims), keepdims, dtype);
        },
        output);
  }
  if (nesteddims.size() > 0) {
    auto opt_sizes = get_opt_sizes(output);
    for (auto opt_size : opt_sizes) {
      TORCH_CHECK(
          opt_size,
          "Current shape doesn't support reduction across nested dimension. Please open a feature request https://t.ly/62F6.");
    }
    auto new_nested_size = get_nested_size(output);
    for (size_t i = nesteddims.size(); i > 0; i--) {
      new_nested_size = squeeze(new_nested_size, nesteddims[i - 1], keepdims);
    }
    auto tmp =
        fn(NestedTensor_to_tensor(output, c10::nullopt),
           IntArrayRef(nesteddims),
           keepdims,
           dtype);
    return wrap_buffer(tmp.reshape({-1}), new_nested_size);
  }
  return output;
}

Tensor NestedTensor_sum_dim(
    const Tensor& self,
    c10::ArrayRef<int64_t> dims,
    bool keepdims,
    c10::optional<ScalarType> dtype) {
  auto my_sum = [](const Tensor& self,
                   IntArrayRef dims,
                   bool keepdims,
                   c10::optional<ScalarType> dtype) {
    return at::sum(self, dims, keepdims, dtype);
  };
  return NestedTensor_func_dim<decltype(my_sum)>(
      my_sum, self, dims, keepdims, dtype);
}

std::tuple<Tensor, Tensor> NestedTensor_max_dim(
    const Tensor& self,
    int64_t dim,
    bool keepdims) {
  int64_t nested_dim = get_nested_tensor_impl(self)->nested_dim();
  at::Tensor output = self;
  if (dim >= nested_dim) {
    std::vector<TensorNode> result = unzip(map(
        [nested_dim, dim, keepdims](at::Tensor tensor) {
          auto tmp = at::max(tensor, dim - nested_dim, keepdims);
          std::vector<at::Tensor> result_i;
          result_i.push_back(std::get<0>(tmp));
          result_i.push_back(std::get<1>(tmp));
          return result_i;
        },
        get_nested_tensor_structure(output)));
    return std::make_tuple(
        wrap_tensor_node(std::move(result[0])),
        wrap_tensor_node(std::move(result[1])));
  }
  auto opt_sizes = get_opt_sizes(output);
  TORCH_CHECK(
      opt_sizes[dim],
      "Current shape doesn't support reduction across nested dimension. Please open a feature request https://t.ly/62F6.");
  auto new_nested_size = get_nested_size(output);
  new_nested_size = squeeze(new_nested_size, dim, keepdims);
  auto tmp =
      at::max(NestedTensor_to_tensor(output, c10::nullopt), dim, keepdims);
  return std::make_tuple(
      wrap_buffer(std::get<0>(tmp).reshape({-1}), new_nested_size),
      wrap_buffer(std::get<1>(tmp).reshape({-1}), new_nested_size));
}

Tensor NestedTensor_max(const Tensor& self) {
  auto tensors = flatten_nested_tensor(map_nested_tensor(
      [](at::Tensor tensor) { return at::max(tensor); }, self));
  if (tensors.size() == 0) {
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors);
  return at::max(all_tensor);
}

Tensor NestedTensor_mean_dim(
    const Tensor& self,
    c10::ArrayRef<int64_t> dims,
    bool keepdims,
    c10::optional<ScalarType> dtype) {
  auto my_mean = [](const Tensor& self,
                    IntArrayRef dims,
                    bool keepdims,
                    c10::optional<ScalarType> dtype) {
    return at::mean(self, dims, keepdims, dtype);
  };
  return NestedTensor_func_dim<decltype(my_mean)>(
      my_mean, self, dims, keepdims, dtype);
}

Tensor NestedTensor_sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten_nested_tensor(map_nested_tensor(
      [&dtype](at::Tensor tensor) { return at::sum(tensor, dtype); }, self));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({0}, *dtype);
    }
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors);
  return at::sum(all_tensor, dtype);
}

Tensor NestedTensor_mean(const Tensor& self, c10::optional<ScalarType> dtype) {
  return at::sum(self, dtype).div_(torch::tensor(get_numel(self)));
}

std::tuple<Tensor, Tensor, Tensor> _make_m2(
    const std::vector<at::Tensor>& tensors,
    IntArrayRef tensordims) {
  std::vector<at::Tensor> m2_tensors;
  std::vector<at::Tensor> mean_tensors;
  std::vector<at::Tensor> numel_tensors;
  for (size_t i = 0; i < tensors.size(); i++) {
    at::Tensor mean = at::mean(tensors[i], tensordims, true);
    at::Tensor centered = tensors[i] - mean;
    m2_tensors.push_back((centered * centered).sum(tensordims, true));
    mean_tensors.push_back(mean);
    int64_t numel = get_numel(tensors[i]) / get_numel(mean);
    numel_tensors.push_back(torch::zeros_like(mean, torch::kLong).fill_(numel));
    // numel_tensors.push_back(torch::tensor({numel}));
  }
  at::Tensor m2_tensor = at::stack(m2_tensors);
  at::Tensor mean_tensor = at::stack(mean_tensors);
  at::Tensor numel_tensor = at::stack(numel_tensors);
  return std::make_tuple(m2_tensor, mean_tensor, numel_tensor);
}

std::tuple<Tensor, Tensor, Tensor> _merge_m2(
    Tensor m2_tensor,
    Tensor mean_tensor,
    Tensor numel) {
  if (m2_tensor.size(0) <= 1) {
    return std::make_tuple(m2_tensor, mean_tensor, numel);
  }
  int64_t start = 0;
  int64_t mid = m2_tensor.size(0) / 2;
  int64_t end = mid * 2;
  at::Tensor numel_0 = at::slice(numel, 0, start, mid);
  at::Tensor numel_1 = at::slice(numel, 0, mid, end);
  at::Tensor mean_0 = at::slice(mean_tensor, 0, start, mid);
  at::Tensor mean_1 = at::slice(mean_tensor, 0, mid, end);
  at::Tensor m2_0 = at::slice(m2_tensor, 0, start, mid);
  at::Tensor m2_1 = at::slice(m2_tensor, 0, mid, end);
  at::Tensor numel_prod = numel_0 * numel_1;
  at::Tensor numel_sum = numel_0 + numel_1;
  at::Tensor delta = mean_0 - mean_1;
  at::Tensor output_m2 =
      (m2_0 + m2_1) + delta * delta * (numel_prod / numel_sum);
  at::Tensor new_mean =
      (numel_0 / numel_sum) * mean_0 + (numel_1 / numel_sum) * mean_1;
  if (end < m2_tensor.size(0)) {
    output_m2 = torch::cat({output_m2, at::slice(m2_tensor, 0, end)});
    new_mean = torch::cat({new_mean, at::slice(mean_tensor, 0, end)});
    numel_sum = torch::cat({numel_sum, at::slice(numel, 0, end)});
  }
  return _merge_m2(output_m2, new_mean, numel_sum);
}

Tensor NestedTensor_var(const Tensor& self, bool unbiased) {
  at::Tensor m2_tensor, mean_tensor, numel;
  std::vector<at::Tensor> tensors = flatten(get_nested_tensor_structure(self));
  if (tensors.size() == 0) {
    return at::ones({0});
  }
  std::vector<int64_t> tensordims;
  for (int64_t i = 0; i < get_dim(tensors[0]); i++) {
    tensordims.push_back(i);
  }
  std::tie(m2_tensor, mean_tensor, numel) =
      _make_m2(tensors, IntArrayRef(tensordims));
  std::tie(m2_tensor, mean_tensor, numel) =
      _merge_m2(m2_tensor, mean_tensor, numel);
  TORCH_CHECK(m2_tensor.size(0) == 1, "output size wrong.");
  if (unbiased) {
    return (m2_tensor / (numel - 1)).reshape({});
  }
  return (m2_tensor / numel).reshape({});
}

Tensor NestedTensor_var_dim(
    const Tensor& self,
    IntArrayRef dims,
    bool unbiased,
    bool keepdims) {
  std::vector<int64_t> tensordims;
  std::vector<int64_t> nesteddims;
  std::tie(tensordims, nesteddims) = make_split_dims(self, dims);

  auto nested_size = get_nested_size(self);
  int64_t nested_dim = get_nested_tensor_impl(self)->nested_dim();
  auto new_nested_size = map(
      [&tensordims](std::vector<int64_t> sizes) {
        std::vector<int64_t> new_sizes;
        for (size_t i = 0; i < sizes.size(); i++) {
          if (std::find(tensordims.begin(), tensordims.end(), i) ==
              tensordims.end()) {
            new_sizes.push_back(sizes[i]);
          }
        }
        return new_sizes;
      },
      nested_size);
  if (nesteddims.size() > 0) {
    TORCH_CHECK(
        nesteddims.size() == 1 && nesteddims[0] == 0,
        "Can only reduce across nested dimension 0.");
    TORCH_CHECK(
        nested_dim == 1,
        "Can only reduce across nested dimensions if given nested tensor is of nested dimension 1.");
    auto opt_sizes = construct_size(new_nested_size);
    for (size_t i = 1; i < opt_sizes.size(); i++) {
      TORCH_CHECK(
          opt_sizes[i],
          "Can only reduce across nested dimensions of Tensor compliant shapes.")
    }
    new_nested_size = squeeze(new_nested_size, 0, keepdims);
  }
  if (tensordims.size() == 0) {
    return wrap_buffer(
        at::var(
            NestedTensor_to_tensor(self, c10::nullopt), 0, unbiased, keepdims)
            .reshape({-1}),
        new_nested_size);
  }
  if (nesteddims.size() == 0) {
    return map_nested_tensor(
        [tensordims, unbiased, keepdims](at::Tensor t) {
          return at::var(t, tensordims, unbiased, keepdims);
        },
        self);
  }

  at::Tensor m2_tensor, mean_tensor, numel;
  std::vector<at::Tensor> tensors = flatten(get_nested_tensor_structure(self));
  std::tie(m2_tensor, mean_tensor, numel) =
      _make_m2(tensors, IntArrayRef(tensordims));
  std::tie(m2_tensor, mean_tensor, numel) =
      _merge_m2(m2_tensor, mean_tensor, numel);
  if (unbiased) {
    return wrap_buffer(
        (m2_tensor / (numel - 1)).reshape({-1}), new_nested_size);
  }
  return wrap_buffer((m2_tensor / numel).reshape({-1}), new_nested_size);
}

Tensor NestedTensor_prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten_nested_tensor(map_nested_tensor(
      [&dtype](at::Tensor tensor) { return at::prod(tensor, dtype); }, self));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({1}, *dtype);
    }
    return at::ones({1});
  }
  auto all_tensor = at::stack(tensors);
  return at::prod(all_tensor, dtype);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "sum", NestedTensor_sum);
  nt_impl(m, "sum.dim_IntList", NestedTensor_sum_dim);
  nt_impl(m, "mean", NestedTensor_mean);
  nt_impl(m, "mean.dim", NestedTensor_mean_dim);
  nt_impl(m, "max", NestedTensor_max);
  nt_impl(m, "max.dim", NestedTensor_max_dim);
  nt_impl(m, "var", NestedTensor_var);
  nt_impl(m, "var.dim", NestedTensor_var_dim);
  nt_impl(m, "prod", NestedTensor_prod);
  nt_impl(m, "cumsum", NestedTensor_cumsum);
}

} // namespace at
