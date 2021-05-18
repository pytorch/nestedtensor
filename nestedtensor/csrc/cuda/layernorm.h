#pragma once
#include <torch/extension.h>
#include <torch/library.h>

namespace torch {
namespace nested_tensor {
namespace cuda {
at::Tensor NestedTensor_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps,
    bool /* cudnn_enable, deprecated */);
}
} // namespace nested_tensor
} // namespace torch
