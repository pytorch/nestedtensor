#include <python_buffer_nested_tensor.h>
#include <python_list_nested_tensor.h>
#include <jit_list_apply.h>

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
  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  py::class_<torch::nested_tensor::THP_BufferNestedTensor>(
      m, "_BufferNestedTensor")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def("get_buffer", &torch::nested_tensor::THP_BufferNestedTensor::get_buffer);


  m.def("jit_apply_function", &torch::nested_tensor::jit_apply_function);
}
