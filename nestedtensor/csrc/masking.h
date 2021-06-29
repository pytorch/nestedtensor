#pragma once
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_functions.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> to_tensor_mask(
    at::Tensor nt,
    c10::optional<int64_t> mask_dim);

at::Tensor to_mask(
    at::Tensor nt,
    c10::optional<int64_t> mask_dim);

at::Tensor to_padded_tensor(
    at::Tensor nt,
    double padding);

at::Tensor from_padded_tensor(
    at::Tensor nt,
    torch::nested_tensor::EfficientSizeNode target_size,
    torch::nested_tensor::EfficientSizeNode target_stride);

at::Tensor from_padded_tensor(
    at::Tensor nt,
    torch::nested_tensor::EfficientSizeNode target_size);

c10::optional<at::Tensor> nt_from_tensor_mask(
    at::Tensor tensor,
    at::Tensor mask,
    int64_t nested_dim);
