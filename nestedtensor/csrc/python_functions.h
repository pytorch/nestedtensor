#pragma once
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

void add_functions(
    pybind11::module,
    pybind11::class_<torch::nested_tensor::THPNestedTensor>);

void add_relu(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);

void add_conv2d(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);

void add_dropout(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);

void add_max_pool2d(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);

void add_batch_norm(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);

void add_cross_entropy(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);

void add_interpolate(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name);
}
} // namespace torch
