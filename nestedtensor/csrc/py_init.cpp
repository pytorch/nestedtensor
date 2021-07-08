#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_functions.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>
#include <chrono>
#include <nestedtensor/csrc/transpose.h>

// NOTE: A NestedTensor without any constituents, i.e.
// nested_tensor([]) is of dimension 1 because
// tensor([]) is of dimension 1, but it is also
// of nested_dim 1, since there are no constituents
// and thus we choose that to imply that the constituents
// tensor dimension is 0.
// If depth is 0, it means that the current structure
// is already a leaf, i.e. has no children.

namespace py = pybind11;

using namespace torch::nested_tensor;
using namespace at;

Tensor get_item(Tensor tensor, py::none key) {
  std::vector<TensorNode> result_nodes;
  result_nodes.push_back(get_nested_tensor_structure(tensor));
  return wrap_tensor_node(TensorNode(std::move(result_nodes)));
}

at::Tensor get_item(Tensor tensor, int64_t key_) {
  std::vector<at::Tensor> unbound = unbind(tensor, 0);
  int64_t key = at::maybe_wrap_dim(key_, unbound.size());
  return unbound[key];
}

#if (PYBIND11_VERSION_MAJOR >= 2 && PYBIND11_VERSION_MINOR >= 3)
at::Tensor get_item(Tensor tensor, py::slice slice) {
  size_t start, stop, step, slicelength;
  if (!slice.compute(nt_size(tensor, 0), &start, &stop, &step, &slicelength)) {
    throw py::error_already_set();
  }
  return at::slice(tensor, 0, start, stop, step);
}

at::Tensor get_item(Tensor tensor, std::vector<py::object> key) {
  if (key.size() == 0) {
    return tensor;
  }
  c10::optional<at::Tensor> first;
  if (!is_nested_tensor_impl(tensor)) {
    auto wrapped_key = py::tuple(py::cast(key));
    auto wrapped_tensor = THPVariable_Wrap(tensor);
    auto wrapped_result =
        torch::autograd::THPVariable_getitem(wrapped_tensor, wrapped_key.ptr());
    auto result = THPVariable_Unpack(wrapped_result);
    Py_DECREF(wrapped_tensor);
    Py_DECREF(wrapped_result);
    return result;
  }
  std::vector<py::object> rest;
  for (size_t i = 1; i < key.size(); i++) {
    rest.push_back(key[i]);
  }
  if (is_nested_tensor_impl(tensor) && py::isinstance<py::none>(key[0])) {
    first = get_item(tensor, py::cast<py::none>(key[0]));
  }
  if (is_nested_tensor_impl(tensor) && py::isinstance<py::int_>(key[0])) {
    first = get_item(tensor, py::cast<int64_t>(key[0]));
  }
  if (is_nested_tensor_impl(tensor) && py::isinstance<py::slice>(key[0])) {
    first = get_item(tensor, py::cast<py::slice>(key[0]));
  }
  TORCH_CHECK(
      first,
      "First entry of tuple doesn't have accepted type. ",
      py::str(key[0]));
  if (!is_nested_tensor_impl(*first)) {
    return get_item(*first, rest);
  }
  std::vector<at::Tensor> result;
  for (auto t : (*first).unbind()) {
    result.push_back(get_item(t, rest));
  }
  int64_t nested_dim = get_nested_tensor_impl(*first)->nested_dim();
  std::vector<TensorNode> result_nodes;
  if (nested_dim == 1) {
    for (auto t : result) {
      result_nodes.push_back(TensorNode(std::move(t)));
    }
  } else {
    for (auto t : result) {
      result_nodes.push_back(get_nested_tensor_structure(t));
    }
  }
  return wrap_tensor_node(TensorNode(std::move(result_nodes)));
}

at::Tensor get_item(Tensor tensor, py::tuple key) {
  std::vector<py::object> entries;
  for (size_t i = 0; i < key.size(); i++) {
    entries.push_back(key[i]);
  }
  auto result = get_item(tensor, entries);
  return result;
}
#endif

py::object _nested_helper(c10::optional<int64_t> index, SizeNode&& size_node) {
  auto fn = [](auto& self, const SizeNode& s, int64_t dim) -> py::object {
    if (dim == 0) {
      return py::cast(s.degree());
    }
    // List of Tensors
    if (s.height() == 1) {
      std::vector<int64_t> result;
      for (const auto& child : s.unbind()) {
        result.push_back(child.payload()[dim - 1]);
      }
      return py::tuple(py::cast(result));
    }
    std::vector<py::object> result;
    for (const auto& child : s.unbind()) {
      result.emplace_back(self(self, child, dim - 1));
    }
    return py::tuple(py::cast(result));
  };
  return fn(fn, size_node, *index);
}

TORCH_LIBRARY(nestedtensor, m) {
  m.def("is_nested_tensor_impl(Tensor tensor) -> bool");
  m.impl("is_nested_tensor_impl", NestedTensorKey, [](Tensor tensor) {
    return is_nested_tensor_impl(tensor);
  });
  m.impl("is_nested_tensor_impl", c10::DispatchKey::CPU, [](Tensor tensor) {
    return is_nested_tensor_impl(tensor);
  });
  m.impl("is_nested_tensor_impl", c10::DispatchKey::CUDA, [](Tensor tensor) {
    return is_nested_tensor_impl(tensor);
  });

  m.def("nested_dim(Tensor tensor) -> int");
  m.impl("nested_dim", NestedTensorKey, [](Tensor tensor) {
    return get_nested_tensor_impl(tensor)->nested_dim();
  });

  m.def("to_nested_tensor(Tensor tensor, int? dim) -> Tensor");
  m.impl(
      "to_nested_tensor",
      NestedTensorKey,
      [](Tensor tensor, c10::optional<int64_t> dim) {
        return NestedTensor_to_nested_tensor(tensor, dim);
      });
  m.impl(
      "to_nested_tensor",
      c10::DispatchKey::CPU,
      [](Tensor tensor, c10::optional<int64_t> dim) {
        return NestedTensor_to_nested_tensor(tensor, dim);
      });
  m.impl(
      "to_nested_tensor",
      c10::DispatchKey::CUDA,
      [](Tensor tensor, c10::optional<int64_t> dim) {
        return NestedTensor_to_nested_tensor(tensor, dim);
      });

  m.def("sizes(Tensor tensor) -> int?[]");
  m.impl("sizes", NestedTensorKey, [](Tensor tensor) {
    return get_nested_tensor_impl(tensor)->opt_sizes();
  });

  m.def("len(Tensor self) -> int");
  m.impl("len", NestedTensorKey, [](Tensor self) {
    return (int64_t)(get_nested_tensor_structure(self).degree());
  });

  m.def("get_dim(Tensor self) -> int");
  m.impl("get_dim", NestedTensorKey, [](Tensor self) { return get_dim(self); });

  m.def("get_numel(Tensor self) -> int");
  m.impl("get_numel", NestedTensorKey, [](Tensor self) {
    return get_numel(self);
  });

  m.def("get_is_contiguous(Tensor self, MemoryFormat memory_format) -> bool");
  m.impl("get_is_contiguous", NestedTensorKey, [](Tensor self, c10::MemoryFormat memory_format) {
    return get_is_contiguous(self, memory_format);
  });

  m.def("transpose_nhwc_nchw(Tensor self) -> Tensor");
  m.impl("transpose_nhwc_nchw", NestedTensorKey, [](Tensor self) {
    return transpose_nhwc_nchw(self);
  });

  m.def("transpose_nchw_nhwc(Tensor self) -> Tensor");
  m.impl("transpose_nchw_nhwc", NestedTensorKey, [](Tensor self) {
    return transpose_nchw_nhwc(self);
  });

  m.def("make_contiguous(Tensor self) -> Tensor");
  m.impl("make_contiguous", NestedTensorKey, [](Tensor self) {
    return NestedTensor_contiguous(self);
  });

  m.def("to_tensor_list(Tensor tensor) -> Tensor[]");
  m.impl("to_tensor_list", NestedTensorKey, [](Tensor tensor) {
    return flatten_nested_tensor(tensor);
  });

  m.def("to_sparse_csr(Tensor tensor) -> Tensor");
  m.impl("to_sparse_csr", NestedTensorKey, [](Tensor tensor) {
    return NestedTensor_to_sparse_csr(tensor);
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  register_python_nested_node(m);
  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.

  m.def("nested_tensor_impl", &torch::nested_tensor::nested_tensor_impl);

  // Need to overwrite because
  // https://github.com/pytorch/pytorch/blob/09660896c0dd2bec888857300a7be9edb52dd05d/aten/src/ATen/TensorIndexing.h#L480
  // requires sizes() for non Tensor-shape compliant NestedTensors
  // and can't be overwritten since it's not a native function.
  // TODO: Advanced indexing
  // TODO: Tensor-wise select
  // TODO: Tuple support
  m.def("get_item", [](Tensor tensor, py::none key) {
    return get_item(tensor, key);
  });
  m.def("get_item", [](Tensor tensor, int64_t key) {
    return get_item(tensor, key);
  });
#if (PYBIND11_VERSION_MAJOR >= 2 && PYBIND11_VERSION_MINOR >= 3)
  m.def("get_item", [](Tensor tensor, py::slice key) {
    return get_item(tensor, key);
  });
  m.def("get_item", [](Tensor tensor, py::tuple key) {
    return get_item(tensor, key);
  });
#endif

  m.def("nested_size", [](Tensor self, c10::optional<int64_t> index_) {
    if (!index_) {
      return py::cast(THPPythonNode(
          map(
              [](std::vector<int64_t> e) {
                return py::reinterpret_steal<py::object>(
                    THPSize_NewFromSizes(e.size(), e.data()));
              },
              get_nested_size(self)),
          "NestedSize"));
    }
    int64_t index = at::maybe_wrap_dim((*index_), get_dim(self));
    return _nested_helper(index, get_nested_size(self));
  });

  m.def("nested_stride", [](Tensor self, c10::optional<int64_t> index_) {
    if (!index_) {
      return py::cast(THPPythonNode(
          map([](std::vector<int64_t> e)
                  -> py::object { return py::tuple(py::cast(e)); },
              get_nested_stride(self)),
          "NestedStride"));
    }
    int64_t index = at::maybe_wrap_dim((*index_), get_dim(self));
    return _nested_helper(index, get_nested_stride(self));
  });

  add_functions(m);
}
