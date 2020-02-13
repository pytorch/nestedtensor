#include <creation.h>
#include <jit_list_apply.h>
#include <nested_node_functions.h>
#include <torch/extension.h>
#include <unary.h>

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

// NOTE: Implementations _ListNestedTensor and _BufferNestedTensor
// return lists of lists of integers for nested_size and nested_stride
// for now. It's up to the consumer to correct this if required.

using namespace torch::nested_tensor;

template <class C>
void add_thp_node(auto m, std::string name) {
  py::class_<C>(m, name.c_str())
      .def("__str__", &C::str)
      .def("unbind", &C::unbind)
      .def("__repr__", &C::str)
      .def("__len__", &C::len);
}

template <class C, class F>
void add_thp_node(auto m, std::string name, F eq_fn) {
  py::class_<C>(m, name.c_str())
      .def("__str__", &C::str)
      .def("unbind", &C::unbind)
      .def("__repr__", &C::str)
      .def("__len__", &C::len)
      .def("__eq__", eq_fn);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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

  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  auto c =
      py::class_<THPNestedTensor>(m, "NestedTensor")
          .def_property_readonly("dtype", &THPNestedTensor::getDtype)
          .def_property_readonly("layout", &THPNestedTensor::getLayout)
          .def_property_readonly("device", &THPNestedTensor::getDevice)
          .def_property_readonly(
              "requires_grad", &THPNestedTensor::requires_grad)
          .def("__len__", &THPNestedTensor::len)
          .def("element_size", &THPNestedTensor::element_size)
          .def(
              "nested_size", py::overload_cast<>(&THPNestedTensor::nested_size))
          .def(
              "nested_size",
              py::overload_cast<c10::optional<int64_t>>(
                  &THPNestedTensor::nested_size))
          .def(
              "nested_stride",
              py::overload_cast<>(&THPNestedTensor::nested_stride))
          .def(
              "nested_stride",
              py::overload_cast<c10::optional<int64_t>>(
                  &THPNestedTensor::nested_stride))
          .def(
              "__getitem__",
              py::overload_cast<int64_t>(&THPNestedTensor::getitem))
          .def(
              "__getitem__",
              py::overload_cast<py::slice>(&THPNestedTensor::getitem))
          .def("unbind", &THPNestedTensor::unbind)
          .def("size", &THPNestedTensor::size)
          .def("requires_grad_", &THPNestedTensor::requires_grad_)
          .def("numel", &THPNestedTensor::numel)
          .def_property_readonly("grad", &THPNestedTensor::grad)
          .def("detach", &THPNestedTensor::detach)
          .def("dim", &THPNestedTensor::dim)
          .def("pin_memory", &THPNestedTensor::pin_memory)
          .def("nested_dim", &THPNestedTensor::nested_dim)
          .def("is_pinned", &THPNestedTensor::is_pinned)
          .def("is_contiguous", &THPNestedTensor::is_contiguous)
          .def("contiguous", &THPNestedTensor::contiguous)
          .def("to_tensor", &THPNestedTensor::to_tensor)
          .def("to_list", &THPNestedTensor::to_list)
          .def("to_tuple", &THPNestedTensor::to_tuple)
          .def("__str__", &THPNestedTensor::str)
          .def("__repr__", &THPNestedTensor::str);

  add_unary_functions(m, c);

  // NOTE: This is a private function until it is feature complete
  m.def("_jit_tensorwise", &torch::nested_tensor::jit_tensorwise);
  m.def("as_nested_tensor", &torch::nested_tensor::as_nested_tensor);
}
