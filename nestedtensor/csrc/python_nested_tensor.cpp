#include <creation.h>
#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

THPIValueNode THPNestedTensor::nested_size() {
  return THPIValueNode(
      map([](c10::List<int64_t> e) { return c10::IValue(e); },
          _data.is_right() ? _data.right().nested_size()
                           : _data.left().nested_size()),
      "NestedSize");
}
THPIValueNode THPNestedTensor::nested_stride() {
  return THPIValueNode(
      map([](c10::List<int64_t> e) { return c10::IValue(e); },
          _data.is_right() ? _data.right().nested_stride()
                           : _data.left().nested_stride()),
      "NestedStride");
}

THPIValueNode _nested_helper(
    c10::optional<int64_t> index,
    SizeNode&& size_node,
    std::string name) {
  auto fn = [](auto& self, const SizeNode& s, int64_t dim) -> IntegerNode {
    if (dim == 0) {
      return IntegerNode(s.degree());
    }
    if (s.height() == 1) {
      return map(
          [dim](c10::List<int64_t> si) { return si.extract(dim - 1); }, s);
    }
    std::vector<IntegerNode> result;
    for (const auto& child : s.unbind()) {
      result.emplace_back(self(self, child, dim - 1));
    }
    return IntegerNode(std::move(result));
  };
  return THPIValueNode(
      map([](int64_t e) { return c10::IValue(e); }, fn(fn, size_node, *index)),
      name);
}

THPIValueNode THPNestedTensor::nested_size(c10::optional<int64_t> index) {
  if (!index) {
    return nested_size();
  }
  auto dim = _data.is_right() ? _data.right().dim() : _data.left().dim();
  // TODO: Negative dims and slices
  TORCH_CHECK(index < dim, "dim argument out of range.");
  SizeNode size_node = _data.is_right() ? _data.right().nested_size()
                                        : _data.left().nested_size();
  return _nested_helper(index, std::move(size_node), "NestedSize");
}
THPIValueNode THPNestedTensor::nested_stride(c10::optional<int64_t> index) {
  if (!index) {
    return nested_stride();
  }
  // TODO: Negative dims and slices
  auto dim = _data.is_right() ? _data.right().dim() : _data.left().dim();
  TORCH_CHECK(index < dim, "dim argument out of range.");
  SizeNode size_node = _data.is_right() ? _data.right().nested_size()
                                        : _data.left().nested_size();
  return _nested_helper(index, std::move(size_node), "NestedStride");
}

std::string THPNestedTensor::str() {
  auto node = _data.is_right() ? _data.right().get_structure()
                               : _data.left().get_structure();
  return NestedNode___str__(
      node, "nested_tensor", [](c10::IValue payload, const std::string& tabs) {
        std::vector<std::string> tokens = split_str(
            THPUtils_unpackString(
                PyObject_Str(THPVariable_Wrap(payload.toTensor()))),
            "\n");
        std::string result;
        for (size_t i = 0; i < tokens.size(); i++) {
          result = result + tabs + tokens[i];
          if (i < tokens.size() - 1) {
            result = result + "\n";
          }
        }
        return result;
      });
}

py::object THPNestedTensor::getDtype() {
  return py::reinterpret_steal<py::object>(
      torch::autograd::utils::wrap(torch::getDtype(
          _data.is_right() ? _data.right().scalar_type()
                           : _data.left().scalar_type())));
}

py::object THPNestedTensor::getLayout() {
  return py::reinterpret_steal<py::object>(
      torch::autograd::utils::wrap(torch::getLayout(
          _data.is_right() ? _data.right().backend()
                           : _data.left().backend())));
}

py::object THPNestedTensor::getDevice() {
  return torch::jit::toPyObject(
      _data.is_right() ? _data.right().device() : _data.left().device());
}

THPNestedTensor THPNestedTensor::contiguous() {
  if (this->is_contiguous()) {
    return *this;
  }
  auto node = _data.is_right() ? _data.right().get_structure()
                               : _data.left().get_structure();
  return THPNestedTensor(make_contiguous(node));
}

// TODO: Since this returns vies there is no reason not to return a sequence of
// contiguous NestedTensors for a given NestedTensor.
py::object THPNestedTensor::unbind() {
  auto node = _data.is_right() ? _data.right().get_structure()
                               : _data.left().get_structure();
  auto nested_dim =
      _data.is_right() ? _data.right().nested_dim() : _data.left().nested_dim();
  if (nested_dim == 1) {
    std::vector<at::Tensor> result;
    for (const auto& child : node.unbind()) {
      result.push_back(child.payload());
    }
    return py::cast(result);
  } else {
    std::vector<THPNestedTensor> result;
    for (const auto& _child : node.unbind()) {
      auto child = _child;
      result.push_back(THPNestedTensor(
          torch::nested_tensor::NestedTensor(std::move(child))));
    }
    return py::cast(result);
  }
}

} // namespace nested_tensor
} // namespace torch
