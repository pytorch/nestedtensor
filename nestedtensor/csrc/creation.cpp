#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/py_utils.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

using namespace torch::jit;

NestedNode<py::object> py_to_nested_node(py::object&& py_obj) {
  if (py::isinstance<py::list>(py_obj) || py::isinstance<py::tuple>(py_obj)) {
    std::vector<NestedNode<py::object>> result;
    auto py_seq = py::sequence(py_obj);
    for (size_t i = 0; i < py_seq.size(); i++) {
      py::object py_seq_i(py_seq[i]);
      result.emplace_back(py_to_nested_node(std::move(py_seq_i)));
    }
    return NestedNode<py::object>(std::move(result));
  }
  TORCH_CHECK(false, "Currently only supporting a list or tuple of py::object.");
}

bool _verify_variables(
    const int64_t dim,
    const at::Layout& layout,
    const at::Device& device,
    const at::ScalarType& scalar_type,
    bool requires_grad,
    const TensorNode& nested_node,
    bool throw_error = false) {
  constexpr const char* advice =
      ("To form a valid NestedTensor all Tensor / NestedTensor constiuents of the given list must be of the same dimension, layout, device,"
       " scalar type and either all or none require gradients. There many further also only be either NestedTensor  / list / tuple entries in a"
       " given list or Tensor entries. Or put differently, if one entry is a Tensor, so must all the others. If one entry is a "
       " NestedTensor / list / tuple, so must all the others.");
  // The attributes must match across all constiuents
  //
  // The NestedTensor's attributes then become that of its
  // constiuents.
  //
  // data must be a list of Tensors or NestedTensors
  //
  // Attributes:
  //     dim()
  //     layout
  //     device
  //     scalar_type
  //     requires_grad
  //     is_pinned()
  bool valid = true;
  if (nested_node.is_leaf()) {
    const at::Tensor& variable = nested_node.payload();
    // TODO: Add more checks?

    valid = valid && (dim == get_dim(variable));
    if (!valid && throw_error) {
      std::stringstream error;
      error << "Given Tensor / NestedTensor constiuent of dimension ";
      error << get_dim(variable);
      error << " doesn't match another constiuent of dimension ";
      error << dim;
      error << ". ";
      error << advice;
      TORCH_CHECK(false, error.str());
    }
    valid = valid && (layout == variable.layout());
    if (!valid && throw_error) {
      std::stringstream error;
      error << "Given Tensor / NestedTensor constiuent of layout ";
      error << variable.layout();
      error << " doesn't match another constiuent of layout ";
      error << layout;
      error << ". ";
      error << advice;
      TORCH_CHECK(false, error.str());
    }
    valid = valid && (device == variable.device());
    if (!valid && throw_error) {
      std::stringstream error;
      error << "Given Tensor / NestedTensor constiuent of device ";
      error << variable.device();
      error << " doesn't match another constiuent of device ";
      error << device;
      error << ". ";
      TORCH_CHECK(false, error.str());
    }
    valid = valid && (scalar_type == variable.scalar_type());
    if (!valid && throw_error) {
      std::stringstream error;
      error << "Given Tensor / NestedTensor constiuent of scalar type ";
      error << variable.scalar_type();
      error << " doesn't match another constiuent of scalar type ";
      error << scalar_type;
      error << ". ";
      TORCH_CHECK(false, error.str());
    }
    valid = valid && (requires_grad == variable.requires_grad());
    if (!valid && throw_error) {
      std::stringstream error;
      if (variable.requires_grad()) {
        error
            << "Given Tensor / NestedTensor constiuent requires gradient in contrast to another constiuent. ";
      } else {
        error
            << "Given Tensor / NestedTensor constiuent doesnt't requires gradient in contrast to another constiuent. ";
      }
      error << advice;
      TORCH_CHECK(false, error.str());
    }
    // TODO: Checking is_pinned is prohibitively costly. It also shouldn't be
    // required. If making the Tensor contiguous we'll create memory in the
    // usual address space and then require the user to move it over into pinned
    // memory manually. However, if it's not contiguous this special memory
    // location might forbid certain operations unexpectedly. For now we blindly
    // rely on those throwing intelligible error.
  } else {
    // NOTE: Checking height is very cheap, so we should do it first.
    for (size_t i = 1; i < nested_node.degree(); i++) {
      valid = valid &&
          (nested_node.children(i).height() ==
           nested_node.children(i - 1).height());
      if (!valid) {
        if (throw_error) {
          TORCH_CHECK(
              false,
              "The to-be constructed NestedTensor is of inconsistent height.");
        }
        break;
      }
    }
    for (size_t i = 0; i < nested_node.degree(); i++) {
      valid = valid &&
          _verify_variables(
                  dim,
                  layout,
                  device,
                  scalar_type,
                  requires_grad,
                  nested_node.children(i),
                  throw_error);
      if (!valid) {
        break;
      }
    }
  }
  return valid;
}

bool _verify_variables(
    const at::Tensor& first_variable,
    const TensorNode& nested_node,
    bool throw_error = false) {
  const int64_t dim = get_dim(first_variable);
  const at::Layout& layout = first_variable.layout();
  const at::Device& device = first_variable.device();
  const at::ScalarType& scalar_type = first_variable.scalar_type();
  bool requires_grad = first_variable.requires_grad();
  return _verify_variables(
      dim,
      layout,
      device,
      scalar_type,
      requires_grad,
      nested_node,
      throw_error);
}

TensorNode py_to_nested_tensor(const py::object& py_obj) {
  if (py::isinstance<py::sequence>(py_obj)) {
    std::vector<TensorNode> result;
    auto py_seq = py::sequence(py_obj);
    for (size_t i = 0; i < py_seq.size(); i++) {
      const py::object& py_seq_i = py_seq[i];
      TORCH_CHECK(THPVariable_Check(py_seq_i.ptr()),
          "Currently only supporting a sequence of Tensors.");
      at::Tensor tensor = THPVariable_Unpack(py_seq_i.ptr());
      TORCH_CHECK(!is_nested_tensor_impl(tensor),
          "Currently do not support NestedTensor entries.");
      result.emplace_back(TensorNode(std::move(tensor)));
    }
    return TensorNode(std::move(result));
  }
  TORCH_CHECK(false, "Currently only supporting a flat sequence of Tensors.");
}

at::Tensor nested_tensor_impl(
    py::sequence list,
    py::object dtype_,
    py::object device_,
    bool requires_grad,
    bool pin_memory,
    bool channels_last) {
  if (requires_grad) {
    throw std::runtime_error(
        "This version of nestedtensor currently does not support autograd. Please open an issue on https://github.com/pytorch/nestedtensor if you need this.");
  }
  auto dtype = toTypeInferredIValue(dtype_).toScalarType();
  auto device = toTypeInferredIValue(device_).toDevice();
  TensorNode ivalue_structure = py_to_nested_tensor(list);
  if (auto first = get_first_leaf(ivalue_structure)) {
    if (!_verify_variables(*first, ivalue_structure)) {
      _verify_variables(*first, ivalue_structure, true);
    }
  }
  Tensor result = wrap_tensor_node(std::move(ivalue_structure));
  Tensor buffer = get_buffer(result);
  buffer = buffer.to(device, dtype);
  if (pin_memory) {
    buffer = buffer.pin_memory();
  }
  result = wrap_buffer(std::move(buffer), get_efficient_nested_size(result));
  if (channels_last) {
    result = NestedTensor_contiguous(result, c10::MemoryFormat::ChannelsLast);
    return result;
  }
  return result;
}

} // namespace nested_tensor
} // namespace torch
