#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/layernorm.h>
#include <nestedtensor/csrc/cuda/transformer_kernels.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace torch {
namespace nested_tensor {
namespace cuda {

Tensor NestedTensor_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  if (weight && bias) {
    if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(*weight) &&
        !is_nested_tensor_impl(*bias)) {
      auto input_opt_sizes = get_opt_sizes(input);
      if (get_dim(input) == 3 && get_is_contiguous(input) &&
          (*input_opt_sizes[2]) % 32 == 0) {
        at::Tensor input_buffer = get_buffer(input);
        int size2 = (int)(*input_opt_sizes[2]);
        int valid_word_num = (int)(input_buffer.numel() / size2);
        at::Tensor zero_bias = torch::zeros({valid_word_num}, input.options());
        at::Tensor output_buffer = torch::zeros_like(input_buffer);
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        if (input_buffer.dtype() == torch::kFloat16) {
          fastertransformer::layer_norm<c10::Half>(
              input_buffer.data_ptr<c10::Half>(),
              weight->data_ptr<c10::Half>(),
              bias->data_ptr<c10::Half>(),
              (c10::Half)(eps),
              output_buffer.data_ptr<c10::Half>(),
              valid_word_num,
              size2,
              defaultStream);
        }
        if (input_buffer.dtype() == torch::kFloat32) {
          fastertransformer::layer_norm<float>(
              input_buffer.data_ptr<float>(),
              weight->data_ptr<float>(),
              bias->data_ptr<float>(),
              (float)(eps),
              output_buffer.data_ptr<float>(),
              valid_word_num,
              size2,
              defaultStream);
        }
        return wrap_buffer(
            std::move(output_buffer),
            get_efficient_nested_size(input),
            get_efficient_nested_stride(input));
      }
    }
    return map_nested_tensor(
        [normalized_shape, eps](const at::Tensor t, Tensor w, Tensor b) {
          return at::layer_norm(t, normalized_shape, w, b, eps, true);
        },
        input,
        *weight,
        *bias);
  }
  TORCH_CHECK(!weight && !bias, "Either both weight and bias are used or not.");
  return map_nested_tensor(
      [normalized_shape, eps](const at::Tensor t) {
        return at::layer_norm(
            t, normalized_shape, c10::nullopt, c10::nullopt, eps, true);
      },
      input);
}
} // namespace cuda
} // namespace nested_tensor
} // namespace torch
