#include <creation.h>
#include <jit_list_apply.h>
#include <nested_node_functions.h>
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

py::object unbind(THPNestedTensor& self) {
  // FOR BUFFER
  if (self.data().is_right()) {
    auto nt = self.data().right();
    if (self.nested_dim() == 1) {
      return wrap_nested_node(nt.get_structure());
    } else {
      std::vector<int64_t> split_sizes;
      auto sizes = nt.nested_size().unbind();
      auto strides = nt.nested_stride().unbind();
      for (int64_t i = 0; i < self.len(); i++) {
        split_sizes.push_back(size_node_memory(sizes[i], strides[i]));
      }
      std::vector<at::Tensor> buffers = at::split_with_sizes(
          nt.get_buffer(), c10::IntArrayRef(split_sizes), 0);
      std::vector<py::object> result;
      for (int64_t i = 0; i < self.len(); i++) {
        result.push_back(
            py::cast(THPNestedTensor(torch::nested_tensor::_BufferNestedTensor(
                std::move(buffers[i]),
                std::move(sizes[i]),
                std::move(strides[i])))));
      }
      return py::cast(result);
    }
  }

  // FOR LIST
  auto nt = self.data().left();
  if (self.nested_dim() == 1) {
    return wrap_nested_node(nt.get_structure());
  } else {
    std::vector<py::object> result;
    for (const auto& _child : nt.get_structure().unbind()) {
      auto child = _child;
      result.push_back(py::cast(THPNestedTensor(
          torch::nested_tensor::_ListNestedTensor(std::move(child)))));
    }
    return py::cast(result);
  }
};

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
  py::class_<THPNestedTensor>(m, "NestedTensor")
      .def_property_readonly("dtype", &THPNestedTensor::getDtype)
      .def_property_readonly("layout", &THPNestedTensor::getLayout)
      .def_property_readonly("device", &THPNestedTensor::getDevice)
      .def_property_readonly("requires_grad", &THPNestedTensor::requires_grad)
      .def("__len__", &THPNestedTensor::len)
      .def("element_size", &THPNestedTensor::element_size)
      .def("nested_size", py::overload_cast<>(&THPNestedTensor::nested_size))
      .def(
          "nested_size",
          py::overload_cast<c10::optional<int64_t>>(
              &THPNestedTensor::nested_size))
      .def(
          "nested_stride", py::overload_cast<>(&THPNestedTensor::nested_stride))
      .def(
          "nested_stride",
          py::overload_cast<c10::optional<int64_t>>(
              &THPNestedTensor::nested_stride))
      .def(
          "__getitem__",
          [](THPNestedTensor self, int64_t key) {
            py::object unbound_ = unbind(self);
            py::sequence unbound = py::cast<py::sequence>(unbound_);
            return unbound[key];
          })
      .def(
          "__getitem__",
          [](THPNestedTensor self, py::slice key) {
            py::object unbound_ = unbind(self);
            py::sequence unbound = py::cast<py::sequence>(unbound_);
            return unbound[key];
          })
      .def("unbind", [](THPNestedTensor self) { return unbind(self); })
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
      .def("get_buffer", &THPNestedTensor::get_buffer)
      .def("to_tensor", &THPNestedTensor::to_tensor)
      .def("to_list", &THPNestedTensor::to_list)
      .def("to_tuple", &THPNestedTensor::to_tuple)
      .def("__str__", &THPNestedTensor::str)
      .def("__repr__", &THPNestedTensor::str);

  // NOTE: This is a private function until it is feature complete
  m.def("_jit_tensorwise", &torch::nested_tensor::jit_tensorwise);
  m.def("as_nested_tensor", &torch::nested_tensor::as_nested_tensor);
}
