#include <c10/cuda/CUDAStream.h>
#include <nestedtensor/csrc/cuda/cuda_kernels.h>
#include <nestedtensor/csrc/cuda/layernorm.h>
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
    std::cout << "1" << std::endl;
    if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(*weight) &&
        !is_nested_tensor_impl(*bias)) {
      std::cout << "2" << std::endl;
      auto input_opt_sizes = get_opt_sizes(input);
      if (get_dim(input) == 3 && get_is_contiguous(input)) {
        std::cout << "3" << std::endl;
        int64_t size2 = *input_opt_sizes[2];
        at::Tensor input_buffer = get_buffer(input);
        at::Tensor output_buffer = torch::zeros_like(input_buffer);
        at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
        nteffectivetransformer::add_bias_input_layernorm_kernelLauncher(
            output_buffer.data_ptr<float>(),
            input_buffer.data_ptr<float>(),
            input_buffer.data_ptr<float>(), // nullptr
            weight->data_ptr<float>(),
            bias->data_ptr<float>(),
            (int)(input_buffer.numel() / size2),
            (int)(size2),
            defaultStream);
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
