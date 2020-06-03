#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <nestedtensor/csrc/nested_tensor.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_relu(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto self_data = self_impl->_data;
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

  // all tensors are same size
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
    
    return wrap_tensor_node(at::max_pool2d(at::stack(tensors),
                                           kernel_size,
                                           stride,
                                           padding,
                                           dilation,
                                           ceil_mode));
  }

  //
  // special kernel
  //
  
  bool flag = true;
  if (kernel_size[0] == 3 && kernel_size[1] == 3 && 
      stride[0] == 2 && stride[1] == 2 && 
      padding[0] == 1 && padding[1] == 1 &&
      dilation[0] == 1 && dilation[1] == 1) {
    // great
  } else {
    bool flag = false;
  }
  
  if (!flag) {
    std::vector<at::Tensor> tensors;
    for (auto tn : tensor_node.unbind()) {
      tensors.push_back(tn.payload());
    }

    std::vector<at::Tensor> unfolded;
    for (auto t : tensors) { 
      unfolded.push_back(
        torch::nn::functional::unfold(t.unsqueeze(0), torch::nn::functional::UnfoldFuncOptions(kernel_size).padding(padding).stride(stride).dilation(dilation))
      );
    }

    auto cat = at::cat(unfolded, 2);
    auto res = at::max_pool2d(
                   cat,
                   /*kernel size*/{9, 1},
                   /*stride*/{9, 1},
                   /*padding*/ 0,
                   /*dilation*/ 1,
                   /*ceil_mode*/ false);

    std::vector<int64_t> split_sizes;
    for (auto t : unfolded) {
      split_sizes.push_back(t.size(2));
    }

    auto splitted = at::split_with_sizes(res, IntArrayRef(split_sizes), 2);

    std::vector<torch::nested_tensor::TensorNode> tensorNodes;
    for (int i = 0; i < tensors.size(); i++) {
      std::vector<int64_t> sizes;
      sizes.push_back(std::ceil(tensors[i].sizes()[1] / 2.0));
      sizes.push_back(std::ceil(tensors[i].sizes()[2] / 2.0));

      auto folded = torch::nn::functional::fold(splitted[i], torch::nn::functional::FoldFuncOptions(/*output size*/ sizes, 
                                                                                                    /*kernel size*/ {1, 1})
                                                                                                    .stride({1, 1})
                                                                                                    .padding(0)
                                                                                                    .dilation(1));
      
      torch::nested_tensor::TensorNode node = torch::nested_tensor::TensorNode(std::move(folded.squeeze(0)));
      tensorNodes.push_back(node);
    }

    return at::detail::make_tensor<at::NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(torch::nested_tensor::TensorNode(std::move(tensorNodes))).to_nested_tensor(nt.nested_dim() - 1));
  }

  //
  // all other cases
  //
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  m.impl_UNBOXED("conv2d", NestedTensor_conv2d);
  m.impl_UNBOXED("batch_norm", NestedTensor_batch_norm);
  m.impl_UNBOXED("max_pool2d", NestedTensor_max_pool2d);
  m.impl_UNBOXED("relu", NestedTensor_relu);
  m.impl_UNBOXED("relu_", NestedTensor_relu_);
  m.impl_UNBOXED("dropout", NestedTensor_dropout);
  m.impl_UNBOXED("dropout_", NestedTensor_dropout_);
  m.impl_UNBOXED("sum", NestedTensor_sum);
}
}
