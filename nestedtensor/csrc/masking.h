#pragma once
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_functions.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> to_tensor_mask(
    at::Tensor nt,
    c10::optional<int64_t> mask_dim);

at::Tensor to_mask(
    at::Tensor nt,
    c10::optional<int64_t> mask_dim);

c10::optional<at::Tensor> nt_from_tensor_mask(
    at::Tensor tensor,
    at::Tensor mask,
    int64_t nested_dim);
