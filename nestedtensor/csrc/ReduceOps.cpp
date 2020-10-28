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
  dim = maybe_wrap_dim(dim, nt_impl->dim());
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
    int64_t dim = maybe_wrap_dim(dims[i], self.dim());
    if (dim < nested_dim) {
      nesteddims.push_back(dim);
    } else {
      tensordims.push_back(dim - nested_dim);
    }
  }
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
    for (auto dim : nesteddims) {
      if (!keepdims) {
        new_nested_size = squeeze(new_nested_size, dim);
      }
    }
    return wrap_buffer(
        fn(NestedTensor_to_tensor(output, c10::nullopt),
           IntArrayRef(nesteddims),
           keepdims,
           dtype)
            .reshape({-1}),
        new_nested_size);
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
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::sum(tensor, dtype); },
          get_nested_tensor_structure(self)));
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
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::mean(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({0}, *dtype);
    }
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors);
  return at::mean(all_tensor, dtype);
}

Tensor NestedTensor_var(const Tensor& self, bool unbiased) {
  auto m2_tensors = flatten(
      map([unbiased](at::Tensor tensor) { return ((tensor - at::mean(tensor, c10::nullopt)) * (tensor - at::mean(tensor, c10::nullopt))).sum(); },
          get_nested_tensor_structure(self)));
  auto mean_tensors = flatten(
      map([unbiased](at::Tensor tensor) { return at::mean(tensor, c10::nullopt); },
          get_nested_tensor_structure(self)));
  at::Tensor numel = torch::tensor(flatten(
      map([](at::Tensor tensor) { return tensor.numel(); },
          get_nested_tensor_structure(self)))).reshape({-1});
  if (m2_tensors.size() == 0) {
    return at::ones({0});
  }
  at::Tensor m2_tensor = at::stack(m2_tensors).reshape({-1});
  at::Tensor mean_tensor = at::stack(mean_tensors).reshape({-1});
  at::Tensor output_m2 = (m2_tensor[0] + m2_tensor[1]) + 
    ((mean_tensor[0] - mean_tensor[1]) * (mean_tensor[0] - mean_tensor[1])) * ((numel[0] * numel[1]) / (numel[0] + numel[1]));
  at::Tensor output = output_m2 / (numel[0] + numel[1]);
  std::cout << "m2_tensor: " << m2_tensor << std::endl;
  std::cout << "mean_tensor: " << mean_tensor << std::endl;
  std::cout << "numel: " << numel << std::endl;
  return output;
}

Tensor NestedTensor_prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::prod(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({1}, *dtype);
    }
    return at::ones({1});
  }
  auto all_tensor = at::stack(tensors);
  return at::prod(all_tensor, dtype);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  nt_impl(m, "sum", NestedTensor_sum);
  nt_impl(m, "sum.dim_IntList", NestedTensor_sum_dim);
  nt_impl(m, "mean", NestedTensor_mean);
  nt_impl(m, "mean.dim", NestedTensor_mean_dim);
  nt_impl(m, "var", NestedTensor_var);
  nt_impl(m, "prod", NestedTensor_prod);
  nt_impl(m, "cumsum", NestedTensor_cumsum);
}

} // namespace at
