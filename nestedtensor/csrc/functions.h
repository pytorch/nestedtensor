#pragma once
#include <nested_tensor.h>
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

NestedTensor squeeze(NestedTensor input, c10::optional<int64_t> dim,
        c10::optional<NestedTensor> out);

THPNestedTensor relu(const THPNestedTensor input, bool inplace); 
THPNestedTensor relu_out(THPNestedTensor& input);
THPNestedTensor dropout(const THPNestedTensor input, double p, bool training, bool inplace);
THPNestedTensor conv2d(const THPNestedTensor input, 
                       const at::Tensor weight, 
                       c10::optional<at::Tensor> bias, 
                       c10::optional<std::vector<int64_t>> stride,
                       c10::optional<std::vector<int64_t>> padding,
                       c10::optional<std::vector<int64_t>> dilation,
                       c10::optional<int64_t> groups);

THPNestedTensor maxPool2d(const THPNestedTensor input,
                          std::vector<int64_t> kernel_size,
                          c10::optional<std::vector<int64_t>> stride,
                          c10::optional<std::vector<int64_t>> padding,
                          c10::optional<std::vector<int64_t>> dilation,
                          c10::optional<bool> return_indices,
                          c10::optional<bool> ceil_mode);

THPNestedTensor batch_norm(const THPNestedTensor input,
                           const at::Tensor running_mean,
                           const at::Tensor running_var,
                           c10::optional<at::Tensor> weight,
                           c10::optional<at::Tensor> bias,
                           bool training, 
                           double momentum,
                           double eps);

THPNestedTensor cross_entropy(const THPNestedTensor input,
                              const THPNestedTensor target,
                              c10::optional<at::Tensor> weight,
                              c10::optional<bool> size_average, // TODO: use
                              c10::optional<int64_t> ignore_index,
                              c10::optional<bool> reduce, // TODO: use
                              c10::optional<std::string> reduction);

THPNestedTensor interpolate(const THPNestedTensor input,
                            c10::optional<std::vector<int64_t>> size,
                            c10::optional<std::vector<double>> scale_factor,
                            c10::optional<std::string> mode,
                            c10::optional<bool> align_corners);

}
}
