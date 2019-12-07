// #include <pybind11/pybind11.h>
#include "python_nested_tensor.h"
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>
// #include <torch/csrc/autograd/utils/wrap_outputs.h>
// #include <torch/csrc/jit/interpreter.h>
// #include <torch/csrc/jit/script/python_sugared_value.h>
// #include <torch/csrc/nestedtensor/python_nested_tensor.h>
// #include <torch/csrc/utils/cuda_lazy_init.h>
// #include <torch/csrc/utils/python_strings.h>


namespace torch {
namespace nested_tensor {

using namespace at;
using namespace torch::autograd;
using namespace torch::autograd::utils;
using namespace torch::jit;
using namespace torch::jit::script;
namespace py = pybind11;

inline PyObject *wrap_list(std::vector<PyObject *> list) {
  auto r = THPObjectPtr{PyTuple_New(list.size())};
  if (!r)
    throw python_error();
  for (size_t i = 0; i < list.size(); ++i) {
    PyTuple_SET_ITEM(r.get(), i, list[i]);
  }
  return r.release();
}

inline PyObject *wrap_nested_node(_NestedNode nested_node) {
  if (nested_node.is_leaf()) {
    return torch::jit::toPyObject(nested_node.payload()).release().ptr();
  } else {
    std::vector<PyObject *> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.push_back(wrap_nested_node(nested_node.children(i)));
    }
    return wrap_list(result);
  }
}

static std::string _NestedNode___str__(const _NestedNode &nested_node) {
  std::stringstream result;
  if (nested_node.is_leaf()) {
    PyObject *objectsRepresentation =
        PyObject_Str(THPVariable_Wrap(nested_node.payload().toTensor()));
    result << THPUtils_unpackString(objectsRepresentation);
    return result.str();
  } else {
    result << "nested_tensor([";
    result << std::endl;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result << "  ";
      result << _NestedNode___str__(nested_node.children(i));
      result << ",";
      result << std::endl;
    }
    result << "])";
    return result.str();
  }
}

static inline _NestedNode _get_structure(PyObject *tensors) {
  if (THPVariable_Check(tensors)) {
    auto variable = THPVariable_Unpack(tensors);
    return _NestedNode(variable);
  } else {
    std::vector<_NestedNode> meta_nodes;
    Py_ssize_t i, n;
    n = PyObject_Length(tensors);
    PyObject *item;
    if (n < 0) {
      throw python_error();
    }
    for (i = 0; i < n; i++) {
      item = PyList_GetItem(tensors, i);
      _NestedNode node = _get_structure(item);
      meta_nodes.push_back(node);
    }
    return _NestedNode(meta_nodes);
  }
}

static inline torch::autograd::Variable _get_first_tensor(PyObject *tensors) {
  if (THPVariable_Check(tensors)) {
    return THPVariable_Unpack(tensors);
  } else {
    return _get_first_tensor(PyList_GetItem(tensors, 0));
  }
}

struct THP_ListNestedTensor {
  THP_ListNestedTensor() = delete;
  THP_ListNestedTensor(py::list list)
      : _data(_ListNestedTensor(_get_structure(list.ptr()))) {}
  THP_ListNestedTensor(_ListNestedTensor data) : _data(data) {}
  int64_t element_size() { return _data.element_size(); }
  py::object nested_size() {
    return py::reinterpret_steal<py::object>(
        wrap_nested_node(_data.nested_size()));
  }
  py::object nested_stride() {
    return py::reinterpret_steal<py::object>(
        wrap_nested_node(_data.nested_stride()));
  }
  THP_ListNestedTensor to(py::args args, py::kwargs kwargs) {
    auto parsed =
        parse_to_conversion(args.ptr(), kwargs.ptr(), /*allow_copy*/ true);
    auto &device = std::get<0>(parsed);
    auto &scalarType = std::get<1>(parsed);
    auto non_blocking = std::get<2>(parsed);
    auto copy = std::get<3>(parsed);
    auto opt_memory_format = std::get<4>(parsed);
    if (device && device->is_cuda()) {
      torch::utils::cuda_lazy_init();
    }
    if (!device && !scalarType && !copy) {
      return *this;
    } else if (!device) {
      return THP_ListNestedTensor(
          _data.to(scalarType.value(), non_blocking, copy, opt_memory_format));
    } else if (!scalarType) {
      return THP_ListNestedTensor(_data.to(_data.options().device(device),
                                           non_blocking, copy,
                                           opt_memory_format));
    } else {
      return THP_ListNestedTensor(_data.to(device.value(), scalarType.value(),
                                           non_blocking, copy,
                                           opt_memory_format));
    }
  }
  THP_ListNestedTensor pin_memory() {
    return THP_ListNestedTensor(_data.pin_memory());
  }
  THP_ListNestedTensor grad() { return THP_ListNestedTensor(_data.grad()); }
  THP_ListNestedTensor detach() { return THP_ListNestedTensor(_data.detach()); }
  THP_ListNestedTensor requires_grad_(py::bool_ requires_grad) {
    return THP_ListNestedTensor(_data.requires_grad_(requires_grad));
  }
  // ADD
  int64_t nested_dim() { return _data.nested_dim(); }
  int64_t dim() { return _data.dim(); }
  bool is_contiguous() { return _data.is_contiguous(); }
  bool is_pinned() { return _data.is_pinned(); }
  bool requires_grad() { return _data.requires_grad(); }
  int64_t numel() { return _data.numel(); }
  int64_t len() { return _data.__len__(); }
  at::Tensor to_tensor() { return _data.to_tensor(); }
  // NOTE: Don't delete this. repr is an important concept, this
  // implementation is just faulty due to torch.Tensor.__repr__
  // TODO: Assuming that there is no difference in __str__ and __repr__ for
  // torch.Tensor.
  std::string str() { return _NestedNode___str__(_data.get_structure()); }
  py::object getDtype() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getDtype(_data.scalar_type())));
  }
  py::object getLayout() {
    return py::reinterpret_steal<py::object>(
        wrap(torch::getLayout(_data.backend())));
  }
  py::object getDevice() { return toPyObject(_data.device()); }
  _ListNestedTensor data() { return _data; }
  void backward(THP_ListNestedTensor gradient, bool retain_graph,
                bool create_graph) {
    _data.backward(gradient.data(), retain_graph, create_graph);
  }

private:
  _ListNestedTensor _data;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  py::class_<THP_ListNestedTensor>(m, "_ListNestedTensor")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def_property_readonly("dtype", &THP_ListNestedTensor::getDtype)
      .def_property_readonly("layout", &THP_ListNestedTensor::getLayout)
      .def_property_readonly("device", &THP_ListNestedTensor::getDevice)
      .def_property_readonly("requires_grad",
                             &THP_ListNestedTensor::requires_grad)
      .def("unbind",
           [](THP_ListNestedTensor self) {
             std::vector<py::object> result;
             if (self.nested_dim() == 1) {
               for (int64_t i = 0; i < self.len(); i++) {
                 result.push_back(toPyObject(
                     self.data().get_structure().children(i).payload()));
               }
             } else {
               for (int64_t i = 0; i < self.len(); i++) {
                 result.push_back(
                     py::cast(THP_ListNestedTensor(_ListNestedTensor(
                         self.data().get_structure().children(i)))));
               }
             }
             return result;
           })
      .def("backward", &THP_ListNestedTensor::backward)
      .def("element_size", &THP_ListNestedTensor::element_size)
      .def("numel", &THP_ListNestedTensor::numel)
      .def("__len__", &THP_ListNestedTensor::len)
      .def("nested_dim", &THP_ListNestedTensor::nested_dim)
      .def("is_contiguous", &THP_ListNestedTensor::is_contiguous)
      .def("is_pinned", &THP_ListNestedTensor::is_pinned)
      .def("dim", &THP_ListNestedTensor::dim)
      .def("nested_size", &THP_ListNestedTensor::nested_size)
      .def("nested_stride", &THP_ListNestedTensor::nested_stride)
      .def("pin_memory", &THP_ListNestedTensor::pin_memory)
      .def("grad", &THP_ListNestedTensor::grad)
      .def("detach", &THP_ListNestedTensor::detach)
      .def("requires_grad_", &THP_ListNestedTensor::requires_grad_)
      .def("to_tensor", &THP_ListNestedTensor::to_tensor)
      .def("__str__", &THP_ListNestedTensor::str)
      .def("__repr__", &THP_ListNestedTensor::str)
      .def("to", &THP_ListNestedTensor::to);
}
} // namespace nested_tensor
} // namespace torch
