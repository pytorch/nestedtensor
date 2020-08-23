#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

namespace impl {
// def _grad_input_padding(grad_output, input_size, stride, padding,
// kernel_size, dilation=None):
//     if dilation is None:
//         # For backward compatibility
//         warnings.warn("_grad_input_padding 'dilation' argument not provided.
//         Default of 1 is used.") dilation = [1] * len(stride)
//
//     input_size = list(input_size)
//     k = grad_output.dim() - 2
//
//     if len(input_size) == k + 2:
//         input_size = input_size[-k:]
//     if len(input_size) != k:
//         raise ValueError("input_size must have {} elements (got {})"
//                          .format(k + 2, len(input_size)))
//
//     def dim_size(d):
//         return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
//         1
//                 + dilation[d] * (kernel_size[d] - 1))
//
//     min_sizes = [dim_size(d) for d in range(k)]
//     max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
//     for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
//         if size < min_size or size > max_size:
//             raise ValueError(
//                 ("requested an input grad size of {}, but valid sizes range "
//                  "from {} to {} (for a grad_output of {})").format(
//                      input_size, min_sizes, max_sizes,
//                      grad_output.size()[2:]))
//
//     return tuple(input_size[d] - min_sizes[d] for d in range(k))
//
//    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
//                                             padding, kernel_size, dilation)

//    kernel_size = [weight.shape[2]]
std::vector<int64_t> _grad_input_padding(
    at::Tensor grad_output,
    IntArrayRef input_size_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef kernel_size,
    IntArrayRef dilation) {
  std::cout << "grad_output.sizes(): " << grad_output.sizes() << std::endl;
  std::cout << "input_size_: " << input_size_ << std::endl;
  std::cout << "stride: " << stride << std::endl;
  std::cout << "padding: " << stride << std::endl;
  std::cout << "kernel_size: " << kernel_size << std::endl;
  std::cout << "dilation: " << dilation << std::endl;
  int64_t k = grad_output.dim() - 2;
  std::cout << "k: " << k << std::endl;

  std::vector<int64_t> input_size;

  if (input_size_.size() == k + 2) {
    for (int64_t i = 2; i < k + 2; i++) {
      input_size.push_back(input_size_[i]);
    }
    // input_size = input_size[-k:] // TODO
    // TORCH_CHECK(false, "NOT IMPLEMENTED");
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

  // def dim_size(d):
  //     return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 1
  //             + dilation[d] * (kernel_size[d] - 1))

  // min_sizes = [dim_size(d) for d in range(k)]
  // max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
  // for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
  //     if size < min_size or size > max_size:
  //         raise ValueError(
  //             ("requested an input grad size of {}, but valid sizes range "
  //              "from {} to {} (for a grad_output of {})").format(
  //                  input_size, min_sizes, max_sizes,
  //                  grad_output.size()[2:]))
  // return tuple(input_size[d] - min_sizes[d] for d in range(k))
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
} // namespace impl

struct NestedTensorFunction_conv2d
    : torch::autograd::Function<NestedTensorFunction_conv2d> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input_,
      const Tensor& weight_,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups) {
    ctx->saved_data["4"] = stride.vec();
    ctx->saved_data["5"] = padding.vec();
    ctx->saved_data["6"] = groups;
    std::cout << "fw dilation: " << dilation << std::endl;
    ctx->saved_data["7"] = dilation.vec();
    auto autograd_input = map_nested_tensor(
        [](at::Tensor ti) {
          AutoGradMode autogradmode(true);
          auto alias = ti.alias();
          alias.requires_grad_();
          return alias;
        },
        input_);
    at::Tensor weight;
    {
      AutoGradMode autogradmode(true);
      weight = weight_.alias().requires_grad_();
    }
    auto autograd_output = map_nested_tensor(
        [&](at::Tensor t) {
          AutoGradMode autogradmode(true);
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
        autograd_input);
    ctx->saved_data["0"] = weight;
    ctx->saved_data["1"] = bias;
    ctx->saved_data["2"] = autograd_output;
    ctx->saved_data["3"] = autograd_input;

    return map_nested_tensor(
        [](at::Tensor t) { return t.detach(); }, autograd_output);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    auto weight = ctx->saved_data["0"].toTensor();
    auto bias = ctx->saved_data["1"].toOptional<at::Tensor>();
    auto autograd_output = ctx->saved_data["2"].toTensor();
    auto autograd_input = ctx->saved_data["3"].toTensor();

    auto stride = ctx->saved_data["4"].toIntList().vec();
    auto padding = ctx->saved_data["5"].toIntList().vec();
    auto groups = ctx->saved_data["6"].toInt();
    auto dilation = ctx->saved_data["7"].toIntList().vec();
    std::cout << "bw dilation: " << dilation << std::endl;

    auto weight_grad = torch::zeros_like(weight);
    c10::optional<at::Tensor> bias_grad;
    if (bias) {
      bias_grad = torch::zeros_like(*bias);
    }
    bool grad_undefined = false;
    bool weight_grad_undefined = false;
    bool bias_grad_undefined = false;

    at::Tensor grad;
    TORCH_CHECK(grad_output.size() == 1, "not supported 0");
    grad = map_nested_tensor(
        [&](at::Tensor r, at::Tensor i, at::Tensor g) {
          // TODO: Might have to retain graph in many to one settings.
          std::vector<at::Tensor> result;
          if (bias) {
            // >>> input = torch.randn(1,1,3,3, requires_grad=True)
            // >>> weight = torch.randn(1,1,1,2, requires_grad=True)
            // >>> output = F.conv2d(input, weight)
            // >>> grad_output = torch.randn(output.shape)
            // >>> grad_input = torch.autograd.grad(output, input, grad_output)
            // >>> F.grad.conv2d_input(input.shape, weight, grad_output)
            // at::Tensor conv_transpose2d(
            //     const Tensor& input, const Tensor& weight, const Tensor&
            //     bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef
            //     output_padding, int64_t groups, IntArrayRef dilation) {
            std::cout << "has bias with sizes " << (*bias).sizes() << std::endl;
            std::cout << "i.sizes(): " << i.sizes() << std::endl;
            auto i_ = i.unsqueeze(0);
            auto g_ = g.unsqueeze(0);

            std::vector<int64_t> kernel_size{weight.size(2), weight.size(3)};
            auto grad_input_padding = impl::_grad_input_padding(
                g_,
                i_.sizes(),
                IntArrayRef(stride),
                IntArrayRef(padding),
                IntArrayRef(kernel_size),
                IntArrayRef(dilation));
            auto grad_input = at::conv_transpose2d(
                                 g_,
                                 weight,
                                 c10::nullopt, //*bias,
                                 IntArrayRef(stride),
                                 IntArrayRef(padding),
                                 IntArrayRef(grad_input_padding),
                                 groups,
                                 IntArrayRef(dilation))
                                 .squeeze(0);
            result = torch::autograd::grad(
                {r}, {i, weight, *bias}, {g}, c10::nullopt, false, true);
            TORCH_CHECK(at::allclose(grad_input, result[0]), "grad input was computed incorrectly.");
          } else {
            result = torch::autograd::grad(
                {r}, {i, weight}, {g}, c10::nullopt, false, true);
          }
          if (!result[1].defined()) {
            weight_grad_undefined = true;
          } else {
            weight_grad.add_(result[1]);
          }
          if (result[2].defined() && bias) {
            (*bias_grad).add_(result[2]);
          } else {
            bias_grad_undefined = true;
          }
          if (!result[0].defined()) {
            grad_undefined = true;
            // TODO: NestedTensor doesn't support undefined devices yet.
            return torch::ones({0});
          }
          return result[0];
        },
        autograd_output,
        autograd_input,
        grad_output[0]);

    at::Tensor undef;
    return {grad_undefined ? undef : grad,
            weight_grad_undefined ? undef : weight_grad,
            bias_grad_undefined || !bias ? undef : *bias_grad,
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
