#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_relu(const Tensor& self) {
  auto self_data = get_nested_tensor(self);
  if (self_data.is_contiguous()) {
    auto res = torch::nested_tensor::NestedTensor(
        at::relu(*self_data.get_buffer()), self_data.nested_size());
    return at::detail::make_tensor<NestedTensorImpl>(std::move(res));
  }
  auto structure = self_data.get_structure();
  auto res =
      map([&](at::Tensor t) { return at::relu(t.unsqueeze(0)).squeeze(0); },
          structure);
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)));
}

Tensor & NestedTensor_relu_(Tensor & self) {
  auto self_data = get_nested_tensor(self);
  if (self_data.is_contiguous()) {
    at::relu_(*self_data.get_buffer());
    return self;
  }
  auto structure = self_data.get_structure();
  apply([](at::Tensor& t) { at::relu_(t); }, structure);
  return self;
}

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  return wrap_tensor_node(
      map([&](const at::Tensor t) { return at::dropout(t, p, train); },
          get_nested_tensor_structure(input)));
}

Tensor& NestedTensor_dropout_(Tensor& input, double p, bool train) {
  apply(
      [&](at::Tensor t) { return at::dropout_(t, p, train); },
      get_nested_tensor_structure(input));
  return input;
}

Tensor NestedTensor_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  return wrap_tensor_node(map(
      [&weight, &bias, &stride, &padding, &dilation, groups](at::Tensor t) {
        return at::convolution(
                   t.unsqueeze(0),
                   weight,
                   bias,
                   stride,
                   padding,
                   dilation,
                   false,
                   {{0, 0}},
                   groups)
            .squeeze(0);
      },
      get_nested_tensor_structure(input)));
}

Tensor NestedTensor_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  auto self_impl = get_nested_tensor_impl(self);
  auto nt = self_impl->_data;
  auto tensor_node = get_nested_tensor_structure(self);

  if (is_tensor_shape(self)) {
    if (self.is_contiguous()) {
      auto buffer = nt.get_buffer();
      auto tensor = torch::reshape(buffer.value(), self_impl->sizes());

      auto res = at::max_pool2d(tensor,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                ceil_mode);

      return at::detail::make_tensor<NestedTensorImpl>(
        torch::nested_tensor::NestedTensor(std::move(res)).to_nested_tensor(nt.nested_dim() - 1));
    }

    std::vector<at::Tensor> tensors;
    for (auto tn : tensor_node.unbind()) {
      tensors.push_back(tn.payload());
    }
    
    auto res = at::max_pool2d(at::stack(tensors),
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              ceil_mode);

    return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(res)).to_nested_tensor(nt.nested_dim() - 1));
  }

  return wrap_tensor_node(map(
      [&](at::Tensor t) {
        return at::max_pool2d(
                   t.unsqueeze(0),
                   kernel_size,
                   stride,
                   padding,
                   dilation,
                   ceil_mode)
            .squeeze(0);
      },
      get_nested_tensor_structure(self)));
}

Tensor NestedTensor_batch_norm(
    const Tensor& input, const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  return wrap_tensor_node(map(
      [&](at::Tensor t) {
        return at::batch_norm(
                   t.unsqueeze(0),
                   weight,
                   bias,
                   running_mean,
                   running_var,
                   training,
                   momentum,
                   eps,
                   cudnn_enabled)
            .squeeze(0);
      },
      get_nested_tensor_structure(input)));
}

Tensor NestedTensor_sum(const Tensor &self, c10::optional<ScalarType> dtype) {
  auto tensors = flatten(
      map([&dtype](at::Tensor tensor) { return at::sum(tensor, dtype); },
          get_nested_tensor_structure(self)));
  if (tensors.size() == 0) {
    if (dtype) {
      return at::ones({0}, *dtype);
    }
    return at::ones({0});
  }
  auto all_tensor = at::stack(tensors.vec());
  return at::sum(all_tensor, dtype);
}

Tensor NestedTensor_reshape(const Tensor& self, IntArrayRef size) {
  auto self_data = get_nested_tensor(self);
  TORCH_CHECK(int64_t(size.size()) > self_data.nested_dim(), "Reshape cannot include nested dimensions.");
  for (int64_t i = 0; i < self_data.nested_dim(); i++) {
    if (size[0] >= 0) {
      throw std::runtime_error("Cannot reshape explicitly along nested dimension "
          + std::to_string(size[i]));
    }
  }
  int64_t nested_dim = self_data.nested_dim();
  std::vector<int64_t> target_shape;
  for (int64_t i = nested_dim; i < size.size(); i++) {
    target_shape.push_back(size[i]);
  }
  return wrap_tensor_node(
      map([target_shape](const at::Tensor t) { 
        return at::reshape(t, IntArrayRef(target_shape));
      },
      get_nested_tensor_structure(self)));
}

Tensor NestedTensor_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  auto self_data = get_nested_tensor(self);
  auto ndims = self_data.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }
  int64_t nested_dim = self_data.nested_dim();
  TORCH_CHECK(dim0 >= nested_dim
           && dim1 >= nested_dim, "Transposition of nested dimensions is not implemented yet.");
  return wrap_tensor_node(
      map([dim0, dim1, nested_dim](const at::Tensor t) { 
        return at::transpose(t, dim0 - nested_dim, dim1 - nested_dim);
        },
      get_nested_tensor_structure(self)));

}

Tensor NestedTensor_softmax(const Tensor& input, const int64_t dim_, c10::optional<ScalarType> dtype) {
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  auto input_data = get_nested_tensor(input);
  int64_t nested_dim = input_data.nested_dim();
  TORCH_CHECK(dim >= nested_dim, "Cannot apply softmax across nested dimensions");
  return wrap_tensor_node(
      map([dim, nested_dim, dtype](const at::Tensor t) { 
        return at::softmax(t, dim - nested_dim, dtype);
        },
      get_nested_tensor_structure(input)));

  // std::cout << "DNDNDN" << std::endl;
  // auto result = [&]() {
  //   NoNamesGuard guard;
  //   if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
  //       return at::_softmax(input_, dim_, true);
  //   } else {
  //       Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
  //       return at::_softmax(converted, dim_, false);
  //   }
  // }();
  // namedinference::propagate_names(result, input_);
  // return result;
  // return input_;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  m.impl_UNBOXED("conv2d", NestedTensor_conv2d);
  m.impl_UNBOXED("batch_norm", NestedTensor_batch_norm);
  m.impl_UNBOXED("max_pool2d", NestedTensor_max_pool2d);
  m.impl_UNBOXED("relu", NestedTensor_relu);
  m.impl_UNBOXED("relu_", NestedTensor_relu_);
  m.impl_UNBOXED("dropout", NestedTensor_dropout);
  m.impl_UNBOXED("dropout_", NestedTensor_dropout_);
  m.impl_UNBOXED("sum", NestedTensor_sum);
  m.impl_UNBOXED("reshape", NestedTensor_reshape);
  m.impl_UNBOXED("transpose.int", NestedTensor_transpose);
  m.impl_UNBOXED("softmax.int", NestedTensor_softmax);
}
}
