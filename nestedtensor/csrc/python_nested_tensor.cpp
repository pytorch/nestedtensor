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
          _data.nested_size()),
      "NestedSize");
}
THPIValueNode THPNestedTensor::nested_stride() {
  return THPIValueNode(
      map([](c10::List<int64_t> e) { return c10::IValue(e); },
          _data.nested_stride()),
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
  auto dim = _data.dim();
  // TODO: Negative dims and slices
  TORCH_CHECK(index < dim, "dim argument out of range.");
  SizeNode size_node = _data.nested_size();
  return _nested_helper(index, std::move(size_node), "NestedSize");
}
THPIValueNode THPNestedTensor::nested_stride(c10::optional<int64_t> index) {
  if (!index) {
    return nested_stride();
  }
  // TODO: Negative dims and slices
  auto dim = _data.dim();
  TORCH_CHECK(index < dim, "dim argument out of range.");
  SizeNode size_node = _data.nested_size();
  return _nested_helper(index, std::move(size_node), "NestedStride");
}

std::string THPNestedTensor::str() {
  auto node = _data.get_structure();
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
      torch::autograd::utils::wrap(torch::getDtype(_data.scalar_type())));
}

py::object THPNestedTensor::getLayout() {
  return py::reinterpret_steal<py::object>(
      torch::autograd::utils::wrap(torch::getLayout(_data.backend())));
}

py::object THPNestedTensor::getDevice() {
  return torch::jit::toPyObject(_data.device());
}

// TODO: Since this returns vies there is no reason not to return a sequence of
// contiguous NestedTensors for a given NestedTensor.
py::object THPNestedTensor::unbind(int64_t dim) {
  auto node = _data.get_structure();
  auto nested_dim = _data.nested_dim();
  if (nested_dim == 1) {
    if (dim == 0) {
      std::vector<at::Tensor> result;
      for (const auto& child : node.unbind()) {
        result.push_back(child.payload());
      }
      return py::cast(result);
    } else {
      int64_t dim_max_size = 0;
      for (const auto& child : node.unbind()) {
        int64_t dim_size = child.payload().size(dim - 1);
        dim_max_size = dim_max_size > dim_size ? dim_max_size : dim_size;
      }
      std::vector<std::vector<TensorNode>> unbound;
      unbound.resize(dim_max_size);
      for (const auto& child : node.unbind()) {
        std::vector<at::Tensor> unbound_tensors =
            at::unbind(child.payload(), dim - 1);
        for (size_t i = 0; i < unbound_tensors.size(); i++) {
          unbound[i].push_back(TensorNode(std::move(unbound_tensors[i])));
        }
      }
      std::vector<THPNestedTensor> result;
      for (size_t i = 0; i < unbound.size(); i++) {
        TensorNode tmp = TensorNode(std::move(unbound[i]));
        result.push_back(THPNestedTensor(NestedTensor(std::move(tmp))));
      }
      return py::cast(result);
    }
  } else {
    std::vector<THPNestedTensor> result;
    if (dim == 0) {
      for (const auto& _child : node.unbind()) {
        auto child = _child;
        result.push_back(THPNestedTensor(NestedTensor(std::move(child))));
      }
    }
    return py::cast(result);
  }
}

} // namespace nested_tensor
} // namespace torch
