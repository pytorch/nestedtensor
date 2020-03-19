#include <creation.h>
#include <py_utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>
#include <utils/nested_node.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

NestedNode<py::object> py_to_nested_node(py::object&& py_obj) {
  if (py::isinstance<py::list>(py_obj) || py::isinstance<py::tuple>(py_obj)) {
    std::vector<NestedNode<py::object>> result;
    auto py_seq = py::sequence(py_obj);
    for (size_t i = 0; i < py_seq.size(); i++) {
      py::object py_seq_i(py_seq[i]);
      result.emplace_back(py_to_nested_node(std::move(py_seq_i)));
    }
    return NestedNode<py::object>(std::move(result));
  } else {
    return NestedNode<py::object>(std::move(py_obj));
  }
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

    valid = valid && (dim == variable.dim());
    if (!valid && throw_error) {
      std::stringstream error;
      error << "Given Tensor / NestedTensor constiuent of dimension ";
      error << variable.dim();
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
  const int64_t dim = first_variable.dim();
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

NestedNode<c10::IValue> py_to_nested_tensor(const py::object& py_obj) {
  if (py::isinstance<py::sequence>(py_obj)) {
    std::vector<NestedNode<c10::IValue>> result;
    auto py_seq = py::sequence(py_obj);
    for (size_t i = 0; i < py_seq.size(); i++) {
      result.emplace_back(py_to_nested_tensor(py_seq[i]));
    }
    return NestedNode<c10::IValue>(std::move(result));
  } else {
    return NestedNode<c10::IValue>(py_obj_to_ivalue(py_obj));
  }
}

THPNestedTensor as_nested_tensor(py::sequence list) {
  NestedNode<c10::IValue> ivalue_structure = py_to_nested_tensor(list);
  auto fn = [](c10::IValue a, bool result) { return result && a.isTensor(); };
  bool all_same =
      reduce<decltype(fn), bool, c10::IValue>(ivalue_structure, fn, true);
  TORCH_CHECK(
      all_same,
      "Input nested list entries need to consist entirely of Tensors or NestedTensors.");
  TensorNode structure =
      map([](c10::IValue a) { return a.toTensor(); }, ivalue_structure);
  if (auto first = get_first_leaf(structure)) {
    if (!_verify_variables(*first, structure)) {
      _verify_variables(*first, structure, true);
    }
  }
  return THPNestedTensor(NestedTensor(std::move(structure)));
}

} // namespace nested_tensor
} // namespace torch
