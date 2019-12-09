#include <python_list_nested_tensor.h>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;
namespace py = pybind11;

struct THP_ListNestedTensor {
  THP_ListNestedTensor() = delete;
  THP_ListNestedTensor(py::list list)
      : _data(_ListNestedTensor(_get_structure(list.ptr()))) {}
  THP_ListNestedTensor(_ListNestedTensor data) : _data(data) {}
  int64_t element_size() {
    return _data.element_size();
  }
  py::object nested_size() {
    return py::reinterpret_steal<py::object>(
        wrap_nested_node(_data.nested_size()));
  }
  py::object nested_stride() {
    return py::reinterpret_steal<py::object>(
        wrap_nested_node(_data.nested_stride()));
  }
  THP_ListNestedTensor pin_memory() {
    return THP_ListNestedTensor(_data.pin_memory());
  }
  THP_ListNestedTensor grad() {
    return THP_ListNestedTensor(_data.grad());
  }
  THP_ListNestedTensor detach() {
    return THP_ListNestedTensor(_data.detach());
  }
  THP_ListNestedTensor requires_grad_(py::bool_ requires_grad) {
    return THP_ListNestedTensor(_data.requires_grad_(requires_grad));
  }
  // ADD
  int64_t nested_dim() {
    return _data.nested_dim();
  }
  int64_t dim() {
    return _data.dim();
  }
  bool is_contiguous() {
    return _data.is_contiguous();
  }
  bool is_pinned() {
    return _data.is_pinned();
  }
  bool requires_grad() {
    return _data.requires_grad();
  }
  int64_t numel() {
    return _data.numel();
  }
  int64_t len() {
    return _data.__len__();
  }
  at::Tensor to_tensor() {
    return _data.to_tensor();
  }
  // NOTE: Don't delete this. repr is an important concept, this
  // implementation is just faulty due to torch.Tensor.__repr__
  // TODO: Assuming that there is no difference in __str__ and __repr__ for
  // torch.Tensor.
  std::string str() {
    return _NestedNode___str__(_data.get_structure());
  }
  py::object getDtype() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getDtype(_data.scalar_type())));
  }
  py::object getLayout() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getLayout(_data.backend())));
  }
  py::object getDevice() {
    return toPyObject(_data.device());
  }
  _ListNestedTensor data() {
    return _data;
  }
  void backward(
      THP_ListNestedTensor gradient,
      bool retain_graph,
      bool create_graph) {
    _data.backward(gradient.data(), retain_graph, create_graph);
  }

 private:
  _ListNestedTensor _data;
};

static _NestedNode apply_jit_function(
    const std::vector<_NestedNode>& nested_nodes,
    Function& fn) {
  bool all_leaf = true;
  for (size_t i = 0; i < nested_nodes.size(); i++) {
    all_leaf = all_leaf && nested_nodes[i].is_leaf();
  }
  if (all_leaf) {
    // NOTE: Assuming this is a pure function not a method (no self!)
    // NOTE: We assume there is only one Tensor inputs.
    // NOTE: We assume no named tensors and no sparse variables as
    // appropriate
    // for TorchScript. NOTE: We know the IValues of the argument, there is
    // no
    // need to cast around.
    Stack stack;
    stack.reserve(nested_nodes.size());
    for (size_t i = 0; i < nested_nodes.size(); i++) {
      push(stack, nested_nodes[i].payload().toTensor());
    }
    fn.run(stack);
    torch::autograd::Variable result = stack.back().toTensor();
    auto result_node = _NestedNode(result);
    return result_node;
  } else {
    bool broadcastable = true;
    size_t num_children = 0;
    for (size_t i = 0; i < nested_nodes.size(); i++) {
      if (!nested_nodes[i].is_leaf()) {
        if (num_children > 0) {
          broadcastable =
              broadcastable && (num_children == nested_nodes[i].degree());
        } else {
          num_children = nested_nodes[i].degree();
        }
      }
    }
    TORCH_CHECK(broadcastable, "Can't broadcast given nested tensors");
    std::vector<_NestedNode> result;
    for (size_t i = 0; i < num_children; i++) {
      std::vector<_NestedNode> local_args;
      for (size_t j = 0; j < nested_nodes.size(); j++) {
        if (nested_nodes[j].is_leaf()) {
          local_args.push_back(nested_nodes[j]);
        } else {
          local_args.push_back(nested_nodes[j].children(i));
        }
      }
      result.push_back(apply_jit_function(local_args, fn));
    }
    return _NestedNode(result);
  }
}
static THP_ListNestedTensor jit_apply_function(
    std::vector<THP_ListNestedTensor> nts_,
    py::object fn) {
  std::vector<_ListNestedTensor> nts;
  for (size_t i = 0; i < nts_.size(); i++) {
    nts.push_back(nts_[i].data());
  }
  auto sfn = py::cast<StrongFunctionPtr>(fn);
  auto tracing_state = tracer::getTracingState();
  TORCH_CHECK(!tracing_state, "doesnt support tracing");
  Function& callee = *sfn.function_;
  auto schema = callee.getSchema();
  TORCH_CHECK(
      schema.arguments().size() == nts.size(),
      "Give NestedTensors don't match function args.");
  std::vector<_NestedNode> nested_nodes;
  for (size_t i = 0; i < nts.size(); i++) {
    nested_nodes.push_back(nts[i].get_structure());
  }
  py::gil_scoped_release release;
  _NestedNode nested_node = apply_jit_function(nested_nodes, callee);
  py::gil_scoped_acquire acquire;
  return THP_ListNestedTensor(_ListNestedTensor(nested_node));
}

} // namespace nested_tensor
} // namespace torch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  py::class_<torch::nested_tensor::THP_ListNestedTensor>(m, "_ListNestedTensor")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def_property_readonly(
          "dtype", &torch::nested_tensor::THP_ListNestedTensor::getDtype)
      .def_property_readonly(
          "layout", &torch::nested_tensor::THP_ListNestedTensor::getLayout)
      .def_property_readonly(
          "device", &torch::nested_tensor::THP_ListNestedTensor::getDevice)
      .def_property_readonly(
          "requires_grad",
          &torch::nested_tensor::THP_ListNestedTensor::requires_grad)
      .def(
          "unbind",
          [](torch::nested_tensor::THP_ListNestedTensor self) {
            std::vector<py::object> result;
            if (self.nested_dim() == 1) {
              for (int64_t i = 0; i < self.len(); i++) {
                result.push_back(torch::jit::toPyObject(
                    self.data().get_structure().children(i).payload()));
              }
            } else {
              for (int64_t i = 0; i < self.len(); i++) {
                result.push_back(
                    py::cast(torch::nested_tensor::THP_ListNestedTensor(
                        torch::nested_tensor::_ListNestedTensor(
                            self.data().get_structure().children(i)))));
              }
            }
            return result;
          })
      .def("backward", &torch::nested_tensor::THP_ListNestedTensor::backward)
      .def(
          "element_size",
          &torch::nested_tensor::THP_ListNestedTensor::element_size)
      .def("numel", &torch::nested_tensor::THP_ListNestedTensor::numel)
      .def("__len__", &torch::nested_tensor::THP_ListNestedTensor::len)
      .def(
          "nested_dim", &torch::nested_tensor::THP_ListNestedTensor::nested_dim)
      .def(
          "is_contiguous",
          &torch::nested_tensor::THP_ListNestedTensor::is_contiguous)
      .def("is_pinned", &torch::nested_tensor::THP_ListNestedTensor::is_pinned)
      .def("dim", &torch::nested_tensor::THP_ListNestedTensor::dim)
      .def(
          "nested_size",
          &torch::nested_tensor::THP_ListNestedTensor::nested_size)
      .def(
          "nested_stride",
          &torch::nested_tensor::THP_ListNestedTensor::nested_stride)
      .def(
          "pin_memory", &torch::nested_tensor::THP_ListNestedTensor::pin_memory)
      .def("grad", &torch::nested_tensor::THP_ListNestedTensor::grad)
      .def("detach", &torch::nested_tensor::THP_ListNestedTensor::detach)
      .def(
          "requires_grad_",
          &torch::nested_tensor::THP_ListNestedTensor::requires_grad_)
      .def("to_tensor", &torch::nested_tensor::THP_ListNestedTensor::to_tensor)
      .def("__str__", &torch::nested_tensor::THP_ListNestedTensor::str)
      .def("__repr__", &torch::nested_tensor::THP_ListNestedTensor::str);
  // .def("to", &THP_ListNestedTensor::to);
}
