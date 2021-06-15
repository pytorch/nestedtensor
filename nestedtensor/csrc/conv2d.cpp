#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_conv2d(
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  Tensor input = input_;
  if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(weight)) {
    // std::cout << "weight.sizes(): " << weight.sizes() << std::endl;
    // std::cout << "stride: " << stride << std::endl;
    // std::cout << "padding: " << padding << std::endl;
    // std::cout << "dilation: " << dilation << std::endl;
    // std::cout << "groups: " << groups << std::endl;
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        // weight.size(0) == weight.size(1) &&
        groups == 1
      ) {
      input = NestedTensor_contiguous(input);
      auto unbound_input = at::unbind(input, 0);
      std::vector<at::Tensor> unfolded_input;
      std::vector<at::Tensor> result_splits;
      at::Tensor weight_view = weight.view({weight.size(0), -1});
      weight_view = weight_view.transpose(0, 1);
      for(size_t i = 0; i < unbound_input.size(); i++) {
        at::Tensor unfolded_split_i =  torch::im2col(unbound_input[i].unsqueeze(0),
                                                  {weight.size(2), weight.size(3)},
                                                  {1, 1}, {0, 0}, {1, 1});
        unfolded_split_i = unfolded_split_i.transpose(1, 2);
        unfolded_input.push_back(unfolded_split_i);
        at::Tensor result_split_i = at::matmul(unfolded_split_i, weight_view).transpose(1, 2);
        result_splits.push_back(result_split_i.reshape(-1));
      }
      at::Tensor result_buffer = at::cat(result_splits);
      // std::cout << "calling packed conv2d" << std::endl;
      return wrap_buffer(std::move(result_buffer), get_efficient_nested_size(input),
          get_efficient_nested_stride(input));
    }
  }
  if (bias) {
      return map_nested_tensor(
          [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
            return at::conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
            // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
          },
          input,
          weight,
          *bias);
  }
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight) {
        return at::conv2d(input.unsqueeze(0), weight, c10::nullopt, stride, padding, dilation, groups).squeeze(0);
        // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
      },
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
