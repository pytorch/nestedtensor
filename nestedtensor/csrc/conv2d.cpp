#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

namespace impl {
// Transliteration of
// https://github.com/pytorch/pytorch/blob/1f0cfbaaad09921f588adf549751041b8cb2e283/torch/nn/grad.py#L8
// into C++
std::vector<int64_t> _grad_input_padding(
    at::Tensor grad_output,
    IntArrayRef input_size_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef kernel_size,
    IntArrayRef dilation) {
  size_t k = grad_output.dim() - 2;
  std::vector<int64_t> input_size;
  if (input_size_.size() == k + 2) {
    for (int64_t i = 2; i < k + 2; i++) {
      input_size.push_back(input_size_[i]);
    }
  } else {
    input_size = input_size_.vec();
  }
  TORCH_CHECK(
      input_size.size() == k,
      "input_size must have ",
      k + 2,
      " elements (got ",
      input_size_.size(),
      ")");

  std::vector<int64_t> result_size;
  for (int64_t d = 0; d < k; d++) {
    int64_t min_size = ((grad_output.size(d + 2) - 1) * stride[d]) -
        (2 * padding[d]) + 1 + (dilation[d] * (kernel_size[d] - 1));
    int64_t max_size = min_size + stride[d] - 1;
    TORCH_CHECK(
        !(input_size[d] < min_size || input_size[d] > max_size),
        "input grad size outside of valid range. input_size[",
        d,
        "]: ",
        input_size[d],
        " min_size: ",
        min_size,
        " max_size: ",
        max_size);
    result_size.push_back(input_size[d] - min_size);
  }
  return result_size;
}

// Transliteration of
// https://github.com/pytorch/pytorch/blob/1f0cfbaaad09921f588adf549751041b8cb2e283/torch/nn/grad.py#L129
// into C++
at::Tensor _conv2d_grad_input(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  std::vector<int64_t> kernel_size{weight.size(2), weight.size(3)};
  auto grad_input_padding = _grad_input_padding(
      grad_output,
      input.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(kernel_size),
      IntArrayRef(dilation));
  auto grad_input = at::conv_transpose2d(
      grad_output,
      weight,
      c10::nullopt, //*bias,
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(grad_input_padding),
      groups,
      IntArrayRef(dilation));
  return grad_input;
}

// Transliteration of
// https://github.com/pytorch/pytorch/blob/1f0cfbaaad09921f588adf549751041b8cb2e283/torch/nn/grad.py#L170
// into C++
at::Tensor _conv2d_grad_weight(
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  int64_t in_channels = input_.size(1);
  int64_t out_channels = grad_output_.size(1);
  int64_t min_batch = input_.size(0);
  auto weight_size = weight.sizes();
  at::Tensor grad_output =
      grad_output_.contiguous().repeat({1, in_channels / groups, 1, 1});
  grad_output =
      grad_output.contiguous().view({grad_output.size(0) * grad_output.size(1),
                                     1,
                                     grad_output.size(2),
                                     grad_output.size(3)});
  at::Tensor input = input_.contiguous().view(
      {1, input_.size(0) * input_.size(1), input_.size(2), input_.size(3)});
  at::Tensor grad_weight = at::conv2d(
      input,
      grad_output,
      c10::nullopt,
      dilation,
      padding,
      stride,
      in_channels * min_batch);
  grad_weight = grad_weight.contiguous().view({min_batch,
                                               grad_weight.size(1) / min_batch,
                                               grad_weight.size(2),
                                               grad_weight.size(3)});
  return grad_weight.sum(0)
      .view({in_channels / groups,
             out_channels,
             grad_weight.size(2),
             grad_weight.size(3)})
      .transpose(0, 1)
      .narrow(2, 0, weight_size[2])
      .narrow(3, 0, weight_size[3]);
}

} // namespace impl

struct NestedTensorFunction_conv2d
    : torch::autograd::Function<NestedTensorFunction_conv2d> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups) {
    // The final call to .contiguous is of questionable general value
    // but in the context of DETR we'll make it the default.
    at::Tensor output = map_nested_tensor(
        [&](at::Tensor t) {
          return at::conv2d(
                     t.unsqueeze(0),
                     weight,
                     bias,
                     stride,
                     padding,
                     dilation,
                     groups)
              .squeeze(0);
        },
        input);
    at::Tensor undef;
    ctx->save_for_backward({weight, bias ? *bias : undef, output, input});
    ctx->saved_data["4"] = stride.vec();
    ctx->saved_data["5"] = padding.vec();
    ctx->saved_data["6"] = groups;
    ctx->saved_data["7"] = dilation.vec();
    return output;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto saved_data = ctx->get_saved_variables();
    auto weight = saved_data[0];
    c10::optional<at::Tensor> bias;
    if (saved_data[1].defined()) {
      bias = saved_data[1];
    }
    auto autograd_output = saved_data[2];
    auto autograd_input = saved_data[3];

    auto stride = ctx->saved_data["4"].toIntList().vec();
    auto padding = ctx->saved_data["5"].toIntList().vec();
    auto groups = ctx->saved_data["6"].toInt();
    auto dilation = ctx->saved_data["7"].toIntList().vec();

    auto weight_grad = torch::zeros_like(weight);
    c10::optional<at::Tensor> bias_grad;
    if (bias) {
      bias_grad = torch::zeros_like(*bias);
    }

    TORCH_CHECK(grad_output.size() == 1, "not supported 0");
    at::Tensor grad = map_nested_tensor(
        [&](at::Tensor r, at::Tensor i, at::Tensor g) {
          TORCH_CHECK(
              !g.requires_grad(), "conv2d doesn't support double backward.");
          if (bias) {
            (*bias_grad).add_(g.sum(1).sum(1));
          }
          auto i_ = i.unsqueeze(0);
          auto g_ = g.unsqueeze(0);
          weight_grad.add_(impl::_conv2d_grad_weight(
              g_, i_, weight, bias, stride, padding, dilation, groups));
          return impl::_conv2d_grad_input(
                     g_, i_, weight, bias, stride, padding, dilation, groups)
              .squeeze(0);
        },
        autograd_output,
        autograd_input,
        grad_output[0]);
    std::cout << "weight_grad.sum(): " << weight_grad.sum() << std::endl;

    at::Tensor undef;
    return {grad,
            weight_grad,
            bias ? *bias : undef,
            undef,
            undef,
            undef,
            undef,
            undef};
  }
};

Tensor NestedTensor_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  return NestedTensorFunction_conv2d::apply(
      input, weight, bias, stride, padding, dilation, groups);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1_PreAutograd, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
