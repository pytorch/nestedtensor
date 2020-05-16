#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <nestedtensor/csrc/python_functions.h>
#include <torch/csrc/Size.h>
#include <torch/extension.h>

// TODO: Add a field such as is_empty to _NestedNode?
// TODO: Remove Variable-only _NestedNodes and replace them with TensorList?
// TODO: Abstract the common recursive patterns.
// TODO: NestedSize C++ object
// TODO: Align NestedTensor and Tensor C++ API

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

py::object _nested_helper(c10::optional<int64_t> index, SizeNode&& size_node) {
  auto fn = [](auto& self, const SizeNode& s, int64_t dim) -> py::object {
    if (dim == 0) {
      return py::cast(s.degree());
    }
    // List of Tensors
    if (s.height() == 1) {
      std::vector<int64_t> result;
      for (const auto& child : s.unbind()) {
        result.push_back(child.payload().get(dim - 1));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  register_python_nested_node(m);

  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.

  // NOTE: This is a private function until it is feature complete
  m.def("as_nested_tensor_impl", &torch::nested_tensor::as_nested_tensor_impl);
  m.def("is_nested_tensor_impl", [](at::Tensor tensor) {
    return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
  });
  m.def("nested_dim", [](at::Tensor tensor) {
    return get_nested_tensor_impl(tensor)->_data.nested_dim();
  });
  m.def(
      "to_nested_tensor",
      torch::wrap_pybind_function(
          [](at::Tensor tensor, c10::optional<int64_t> dim) {
            auto impl_data = get_nested_tensor_impl(tensor)->_data;
            auto nt = impl_data.to_nested_tensor(dim);
            return at::detail::make_tensor<at::NestedTensorImpl>(std::move(nt));
          }));
  m.def("str", [](at::Tensor tensor) {
    auto impl_data = get_nested_tensor_impl(tensor)->_data;
    auto node = impl_data.get_structure();
    return NestedNode___str__(
        node,
        "nested_tensor",
        [](c10::IValue payload, const std::string& tabs) {
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
  });
  m.def(
      "to_tensor",
      torch::wrap_pybind_function(
          [](at::Tensor tensor, c10::optional<int64_t> dim) {
            return at::NestedTensor_to_tensor(tensor, dim);
          }));
  // Need to overwrite because
  // https://github.com/pytorch/pytorch/blob/09660896c0dd2bec888857300a7be9edb52dd05d/aten/src/ATen/TensorIndexing.h#L480
  // requires sizes() for non Tensor-shape compliant NestedTensors
  // and can't be overwritten since it's not a native function.
  // TODO: Not covered by 0.0.2 or 0.0.1!
  // NOTE: Returns a view
  // TODO: Advanced indexing
  // TODO: Tensor-wise select
  // TODO: Tuple support
  m.def("get_item", [](at::Tensor tensor, int64_t key) {
    return at::unbind(tensor, 0)[key];
  });
#if (PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 4)
  m.def("get_item", [](at::Tensor tensor, py::slice key) {
    py::list unbound = py::cast(at::unbind(tensor, 0));
    return unbound[key];
  });
#endif

  m.def("nested_size", [](at::Tensor self, c10::optional<int64_t> index_) {
    if (!self.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
      throw std::runtime_error("Function requires NestedTensorImpl");
    }
    auto nt =
        static_cast<at::NestedTensorImpl*>(self.unsafeGetTensorImpl())->_data;
    if (!index_) {
      return py::cast(THPPythonNode(
          map(
              [](c10::List<int64_t> e) {
                std::vector<int64_t> e_vec = e.vec();
                return py::reinterpret_steal<py::object>(
                    THPSize_NewFromSizes(e_vec.size(), e_vec.data()));
              },
              nt.nested_size()),
          "NestedSize"));
    }
    int64_t index = at::maybe_wrap_dim((*index_), nt.dim());
    SizeNode size_node = nt.nested_size();
    return _nested_helper(index, std::move(size_node));
  });

  m.def("nested_stride", [](at::Tensor self, c10::optional<int64_t> index_) {
    if (!self.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
      throw std::runtime_error("Function requires NestedTensorImpl");
    }
    auto nt =
        static_cast<at::NestedTensorImpl*>(self.unsafeGetTensorImpl())->_data;
    if (!index_) {
      return py::cast(THPPythonNode(
          map([](c10::List<int64_t> e)
                  -> py::object { return py::tuple(py::cast(e.vec())); },
              nt.nested_stride()),
          "NestedStride"));
    }
    int64_t index = at::maybe_wrap_dim((*index_), nt.dim());
    SizeNode size_node = nt.nested_stride();
    return _nested_helper(index, std::move(size_node));
  });

  m.def("sizes", [](at::Tensor tensor) {
    if (!tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
      throw std::runtime_error("Function requires NestedTensorImpl");
    }
    return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl())
        ->_data.sizes();
  });

  m.def("len", [](at::Tensor self) {
    if (!self.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
      throw std::runtime_error("Function requires NestedTensorImpl");
    }
    auto nt =
        static_cast<at::NestedTensorImpl*>(self.unsafeGetTensorImpl())->_data;
    return nt.get_structure().degree();
  });

  m.def("make_nested_tensor_impl", [](std::vector<at::Tensor> tensors) {
    std::vector<TensorNode> tensor_nodes;
    for (size_t i = 0; i < tensors.size(); i++) {
      tensor_nodes.push_back(TensorNode(std::move(tensors[i])));
    }
    return at::detail::make_tensor<at::NestedTensorImpl>(
        NestedTensor(TensorNode(std::move(tensor_nodes))));
  });

  add_functions(m);
}
