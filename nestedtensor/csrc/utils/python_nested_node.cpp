#include <creation.h>
#include <nested_node_functions.h>
#include <python_nested_node.h>

namespace torch {
namespace nested_tensor {

using THPSizeNode = THPNestedNode<c10::List<int64_t>>;
using THPIntegerNode = THPNestedNode<int64_t>;
using THPTensorNode = THPNestedNode<at::Tensor>;
using THPIValueNode = THPNestedNode<c10::IValue>;
using THPPythonNode = THPNestedNode<py::object>;

using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;
using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using PythonNode = NestedNode<py::object>;

using namespace torch::nested_tensor;
namespace py = pybind11;

template <class C>
void add_thp_node(py::module m, std::string name) {
  py::class_<C>(m, name.c_str())
      .def("__str__", &C::str)
      .def("unbind", &C::unbind)
      .def("__repr__", &C::str)
      .def("__len__", &C::len);
}

template <class C, class F>
void add_thp_node(py::module m, std::string name, F eq_fn) {
  py::class_<C>(m, name.c_str())
      .def("__str__", &C::str)
      .def("unbind", &C::unbind)
      .def("__repr__", &C::str)
      .def("__len__", &C::len)
      .def("__eq__", eq_fn);
}

THPPythonNode as_nested_node(py::sequence _list) {
  py::object list = _list;
  NestedNode<py::object> py_nested_node = py_to_nested_node(std::move(list));
  return THPPythonNode(py_nested_node, "PythonNode");
}

THPPythonNode py_map(py::function fn, THPPythonNode node) {
  PythonNode result =
      map([&fn](py::object obj) { return fn(obj); }, node.get_node());
  return THPPythonNode(result, "PythonNode");
}

void register_python_nested_node(py::module m) {
  add_thp_node<THPPythonNode>(m, "PythonNode");

  add_thp_node<THPSizeNode>(
      m, "SizeNode", [](THPSizeNode& a_, THPSizeNode& b_) {
        SizeNode a = a_.get_node();
        SizeNode b = b_.get_node();
        if (!shape_matches(a, b)) {
          return false;
        }
        auto fn = [](c10::List<int64_t> a, c10::List<int64_t> b) {
          for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {
              return false;
            }
          }
          return true;
        };
        return all<decltype(fn)>(std::move(fn), a, b);
      });

  add_thp_node<THPIValueNode>(
      m, "IValueNode", [](THPIValueNode& a_, THPIValueNode& b_) {
        auto a = a_.get_node();
        auto b = b_.get_node();
        if (!shape_matches(a, b)) {
          return false;
        }
        auto fn1 = [](auto i, auto j) { return (*i.type()) == (*j.type()); };
        if (!all<decltype(fn1)>(std::move(fn1), a, b)) {
          return false;
        }
        auto fn2 = [](auto a, auto b) {
          if (a.isInt()) {
            return a.toInt() == b.toInt();
          }
          if (a.isIntList()) {
            auto a_ = a.toIntList();
            auto b_ = b.toIntList();
            for (size_t i = 0; i < a_.size(); i++) {
              if (a_[i] != b_[i]) {
                return false;
              }
            }
            return true;
          }
          TORCH_CHECK(false, "Type not supported for comparison.");
        };
        return all<decltype(fn2)>(std::move(fn2), a, b);
      });

  m.def("as_nested_node", &as_nested_node);
  m.def("map", &py_map);
}

} // namespace nested_tensor
} // namespace torch
