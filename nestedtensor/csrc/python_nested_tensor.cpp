#include <creation.h>
#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <ATen/WrapDimUtils.h>

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
std::vector<py::object> THPNestedTensor::unbind(int64_t dim) {
  dim = at::maybe_wrap_dim(dim, _data.dim());
  auto node = _data.get_structure();
  auto nested_dim = _data.nested_dim();
  if (nested_dim == 1) {
    if (dim == 0) {
      std::vector<py::object> result;
      for (const auto& child : node.unbind()) {
        result.push_back(py::cast(child.payload()));
      }
      return result;
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
      std::vector<py::object> result;
      for (size_t i = 0; i < unbound.size(); i++) {
        TensorNode tmp = TensorNode(std::move(unbound[i]));
        THPNestedTensor tmp_thd = THPNestedTensor(NestedTensor(std::move(tmp)));
        result.push_back(py::cast(tmp_thd));
      }
      return result;
    }
  }
  std::vector<THPNestedTensor> unbound_thp;
  for (auto child : node.unbind()) {
    unbound_thp.push_back(THPNestedTensor(NestedTensor(std::move(child))));
  }
  if (dim == 0) {
    std::vector<py::object> result;
    for (auto elem : unbound_thp) {
      result.push_back(py::cast(elem));
    }
    return result;
  }
  std::vector<std::vector<TensorNode>> unbound;
  for (size_t i = 0; i < unbound_thp.size(); i++) {
    std::vector<py::object> tmp = unbound_thp[i].unbind(dim - 1);
    for (size_t j = 0; j < tmp.size(); j++) {
      if (unbound.size() >= j) {
        unbound.resize(j + 1);
      }
      py::object tmp_j = tmp[j];
      if (py::isinstance<THPNestedTensor>(tmp_j)) {
        unbound[j].push_back(
            py::cast<THPNestedTensor>(tmp[j])._data.get_structure());
      } else {
        unbound[j].push_back(TensorNode(py::cast<at::Tensor>(tmp_j)));
      }
    }
  }
  std::vector<py::object> result;
  for (size_t i = 0; i < unbound.size(); i++) {
    TensorNode tmp = TensorNode(std::move(unbound[i]));
    THPNestedTensor tmp_thd = THPNestedTensor(NestedTensor(std::move(tmp)));
    result.push_back(py::cast(tmp_thd));
  }
  return result;
}

} // namespace nested_tensor
} // namespace torch
