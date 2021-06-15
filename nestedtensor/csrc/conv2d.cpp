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
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1
      ) {
      input = NestedTensor_contiguous(input);
      auto unbound_input = at::unbind(input, 0);
      std::vector<at::Tensor> unfolded_input;
      std::vector<at::Tensor> result_splits;
      at::Tensor weight_view = weight.view({weight.size(0), -1});
      weight_view = weight_view.transpose(0, 1);
      for(size_t i = 0; i < unbound_input.size(); i++) {
        // std::cout << "unbound_input[" << i << "]: " << unbound_input[i] << std::endl;
        at::Tensor unfolded_split_i =  torch::im2col(unbound_input[i].unsqueeze(0),
                                                  {weight.size(2), weight.size(3)},
                                                  {1, 1}, {0, 0}, {1, 1});
        unfolded_split_i = unfolded_split_i.transpose(1, 2);
        unfolded_input.push_back(unfolded_split_i);
        // std::cout << "unfolded_split_i[" << i << "]: " << unfolded_split_i << std::endl;
        at::Tensor result_split_i = at::matmul(unfolded_split_i, weight_view).transpose(1, 2);
        // std::cout << "result_split_" << i << ": " << result_split_i << std::endl;
        result_splits.push_back(result_split_i.reshape(-1));
      }
      // at::Tensor catted_input = at::cat(unfolded_input, 1);
      // // std::cout << "catted_input: " << catted_input << std::endl;
      // at::Tensor composite_result_buffer = at::matmul(catted_input, weight_view).transpose(1, 2);
      // std::cout << "composite_result_buffer: " << composite_result_buffer << std::endl;
      // std::cout << "composite_result_buffer.reshape(-1): " << composite_result_buffer.reshape(-1) << std::endl;
      // std::cout << "composite_result_buffer.contiguous().reshape(-1): " << composite_result_buffer.contiguous().reshape(-1) << std::endl;
      at::Tensor result_buffer = at::cat(result_splits);
      // std::cout << "result_buffer: " << result_buffer << std::endl;
      return wrap_buffer(std::move(result_buffer), get_efficient_nested_size(input),
          get_efficient_nested_stride(input));
      // return wrap_buffer(composite_result_buffer.reshape(-1), get_efficient_nested_size(input),
      //     get_efficient_nested_stride(input));

      // // auto input_opt_sizes = get_opt_sizes(input);
      // // at::Tensor input_buffer = get_buffer(input);
      // // at::Tensor weight_view = weight.view({weight.size(0), -1});
      // // std::cout << "orig input_buffer: " << input_buffer << std::endl;
      // // input_buffer = input_buffer.reshape({1, 1, -1, weight_view.size(1)});
      // // std::cout << "input_buffer.sizes(): " << input_buffer.sizes() << std::endl;
      // // std::cout << "input_buffer: " << input_buffer << std::endl;
      // // at::Tensor result = at::matmul(input_buffer, weight_view.transpose(0, 1));
      // // std::cout << 
      // //   "at::matmul(input_buffer, weight_view.transpose(0, 1)): " <<
      // //   result <<
      // //   std::endl;
      // // auto ef_size = get_efficient_nested_size(input).sizes();
      // // auto ef_stride = get_efficient_nested_stride(input);
      // // std::cout << "ef_size: " << ef_size << std::endl;
      // // Tensor ef_size0 = at::native::narrow(ef_size, 1, 0, 1);
      // // Tensor ef_size1 = at::native::narrow(ef_size, 1, 1, 1);
      // // Tensor ef_size2 = at::native::narrow(ef_size, 1, 2, 1);
      // // Tensor ef_sizeall = ef_size0 * ef_size1 * ef_size2;
      // // std::cout << "ef_sizeall: " << ef_sizeall << std::endl;
      // // std::vector<int64_t> ef_splits;
      // // for (int64_t i = 0; i < ef_sizeall.size(0); i++) {
      // //   ef_splits.push_back(ef_sizeall[i].item<int64_t>());
      // // }
      // // std::vector<at::Tensor> splits = at::split_with_sizes(input_buffer.reshape({-1}), ef_splits);
      // // std::vector<at::Tensor> unfolded_splits;
      // // for (size_t i = 0; i < splits.size(); i++) {
      // //   splits[i] = splits[i].reshape({-1, weight.size(1), ef_size1[i].item<int64_t>(), ef_size2[i].item<int64_t>()});
      // //   std::cout << "splits[" << i << "]" << splits[i] << std::endl;
      // //   at::Tensor unfolded_split_i =  torch::im2col(splits[i], {weight.size(2), weight.size(3)},
      // //                                               {1, 1}, {0, 0}, {1, 1});
      // //   std::cout << "unfolded_split_" << i << ": " << unfolded_split_i << std::endl;
      // //   unfolded_splits.push_back(unfolded_split_i);
      // // }
      // // at::Tensor unfolded_input = at::cat(unfolded_splits, 2);
      // // std::cout << "unfolded_input: " << unfolded_input << std::endl;
      // // std::cout << "unfolded_input.transpose(1, 2): " << unfolded_input.transpose(1, 2) << std::endl;
      // // at::Tensor result_buffer = at::matmul(unfolded_input.transpose(1, 2), weight.reshape({-1, weight.size(1)}));
      // // std::cout << "result_buffer: " << result_buffer << std::endl;
      // // result_buffer = result_buffer.contiguous();
      // // std::vector<at::Tensor> result_splits = at::split_with_sizes(result_buffer.reshape({-1}), ef_splits);
      // // for (size_t i = 0; i < splits.size(); i++) {
      // //   result_splits[i] = result_splits[i].reshape(-1);
      // //   std::cout << "result_splits[" << i << "]: " << result_splits[i] << std::endl;
      // // }
      // // at::Tensor cat_splits = at::cat(result_splits);
      // // cat_splits = cat_splits.reshape({weight.size(1), -1}).transpose(0, 1).contiguous();
      // // return wrap_buffer(cat_splits.reshape(-1), get_efficient_nested_size(input),
      // //     get_efficient_nested_stride(input));
      // std::cout << "result_buffer: " << result_buffer << std::endl;
      // std::cout << "unfolded_input.sizes(): " << unfolded_input.sizes() << std::endl;
      // at::Tensor folded_result_buffer = at::col2im(result_buffer, {5, 16}, {1, 1}, {1, 1}, {0, 0}, {1, 1});
      // std::cout << "folded_result_buffer: " << folded_result_buffer << std::endl;
      // return wrap_buffer(result.reshape(-1), 
      //     ef_size, ef_stride);

      // std::cout << 
      // "at::conv2d(input_buffer, weight, bias, stride, padding, dilation, groups): " <<
      // at::conv2d(input_buffer, weight.view(1, -1), bias, stride, padding, dilation, groups) <<
      // std::endl;
      // std::cout << 
      //   "torch::im2col(input_buffer, {weight.size(0), weight.size(1)}): " <<
      //   torch::im2col(input_buffer, {weight.size(2), weight.size(3)},
      //       {1, 1}, {0, 0}, {1, 1}) << std::endl;
      // std::cout << "input_buffer.reshape({-1, weight_view.size(0)}): " <<
      //   input_buffer.reshape({1, -1, weight_view.size(1)}).transpose(1, 2) << std::endl;
      // std::cout << "weight: " << weight << std::endl;
      // std::cout << "weight_view: " << weight_view << std::endl;
      // std::cout << "bias.has_value(): " << bias.has_value() << std::endl;
      // std::cout << "is_nested_tensor_impl(input): " << is_nested_tensor_impl(input) << std::endl;
      // std::cout << "is_nested_tensor_impl(weight): " << is_nested_tensor_impl(weight) << std::endl;
      // std::cout << "weight.sizes(): " << weight.sizes() << std::endl;
      // std::cout << "stride: " << stride << std::endl;
      // std::cout << "padding: " << padding << std::endl;
      // std::cout << "dilation: " << dilation << std::endl;
      // std::cout << "groups: " << groups << std::endl;
      // at::Tensor result = at::matmul(weight_view, input_buffer.reshape({1, -1, weight_view.size(1)}).transpose(1, 2));
      // std::cout << "result: " << result << std::endl;
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
