#pragma once
#include <nested_tensor.h>
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

NestedTensor squeeze(NestedTensor input, c10::optional<int64_t> dim,
        c10::optional<NestedTensor> out);

NestedTensor relu(NestedTensor input, c10::optional<bool> inplace); 
NestedTensor relu_out(NestedTensor& input);
NestedTensor dropout(NestedTensor input, 
                     c10::optional<double> p, 
                     c10::optional<bool> training, 
                     c10::optional<bool> inplace);
NestedTensor conv2d(NestedTensor input, 
                    const at::Tensor weight, 
                    c10::optional<at::Tensor> bias, 
                    c10::optional<std::vector<int64_t>> stride,
                    c10::optional<std::vector<int64_t>> padding,
                    c10::optional<std::vector<int64_t>> dilation,
                    c10::optional<int64_t> groups);

NestedTensor max_pool2d(NestedTensor input,
                        at::IntArrayRef kernel_size,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        at::IntArrayRef dilation,
                        bool ceil_mode);

NestedTensor batch_norm(NestedTensor input,
                        const at::Tensor running_mean,
                        const at::Tensor running_var,
                        c10::optional<at::Tensor> weight,
                        c10::optional<at::Tensor> bias,
                        bool training, 
                        double momentum,
                        double eps);

NestedTensor cross_entropy(NestedTensor input,
                           NestedTensor target,
                           c10::optional<at::Tensor> weight,
                           c10::optional<bool> size_average, // TODO: use
                           c10::optional<int64_t> ignore_index,
                           c10::optional<bool> reduce, // TODO: use
                           c10::optional<std::string> reduction);

NestedTensor interpolate(NestedTensor input,
                         c10::optional<std::vector<int64_t>> size,
                         c10::optional<std::vector<double>> scale_factor,
                         c10::optional<std::string> mode,
                         c10::optional<bool> align_corners);

}
}
