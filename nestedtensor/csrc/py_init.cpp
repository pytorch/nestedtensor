#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <nestedtensor/csrc/python_functions.h>
#include <torch/csrc/Size.h>
#include <torch/extension.h>

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


namespace torch {
namespace nested_tensor {
namespace {

static auto registry =
    torch::RegisterOperators()
      .op("nestedtensor::is_nested_tensor_impl", [](Tensor tensor) {
        return is_nested_tensor_impl(tensor);
      })
      .op("nestedtensor::nested_dim", [](Tensor tensor) {
        return get_nested_tensor(tensor).nested_dim();
      })
      .op("nestedtensor::to_nested_tensor",
              [](Tensor tensor, c10::optional<int64_t> dim) {
                auto nt = get_nested_tensor(tensor);
                return wrap_nested_tensor(nt.to_nested_tensor(dim));
              })
      .op("nestedtensor::grad", 
            [](Tensor tensor) {
              auto nt = get_nested_tensor(tensor);
              return wrap_nested_tensor(nt.grad());
            })
      .op("nestedtensor::requires_grad", 
            [](Tensor tensor) {
              auto nt = get_nested_tensor(tensor);
              return nt.requires_grad();
            })
      .op("nestedtensor::requires_grad_",
            [](Tensor tensor, bool requires_grad) {
            auto nt = get_nested_tensor(tensor);
            return wrap_nested_tensor(nt.requires_grad_(requires_grad));
          })
      .op("nestedtensor::backward",
          [](Tensor tensor,
             Tensor gradient,
             bool retain_graph,
             bool create_graph) {
            auto nt = get_nested_tensor(tensor);
            nt.backward(get_nested_tensor(gradient), retain_graph, create_graph);
          })
      .op("nestedtensor::sizes", [](Tensor tensor) {
        return get_nested_tensor(tensor).sizes();
      })
      .op("nestedtensor::len", [](Tensor self) {
        return (int64_t)(get_nested_tensor(self).get_structure().degree());
      })
      .op("nestedtensor::to_tensor",
        [](Tensor tensor, c10::optional<int64_t> dim) {
          return NestedTensor_to_tensor(tensor, dim);
        })
      .op("nestedtensor::str", [](Tensor tensor) {
        auto node = get_nested_tensor(tensor).get_structure();
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
        })
      ;

}
} // namespace nested_tensor
} // namespace torch

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
  m.def("get_item", [](Tensor tensor, int64_t key_) {
    std::vector<at::Tensor> unbound = unbind(tensor, 0);
    int64_t key = at::maybe_wrap_dim(key_, unbound.size());
    return unbind(tensor, 0)[key];
  });
#if (PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 4)
  m.def("get_item", [](Tensor tensor, py::slice key) {
    py::list unbound = py::cast(unbind(tensor, 0));
    return unbound[key];
  });
#endif

  m.def("nested_size", [](Tensor self, c10::optional<int64_t> index_) {
    auto nt = get_nested_tensor(self);
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

  m.def("nested_stride", [](Tensor self, c10::optional<int64_t> index_) {
    auto nt = get_nested_tensor(self);
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

  add_functions(m);
}

