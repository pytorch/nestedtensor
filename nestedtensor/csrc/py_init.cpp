#include <creation.h>
#include <jit_list_apply.h>
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<torch::nested_tensor::THPSizeNode>(m, "SizeNode")
      .def("__str__", &torch::nested_tensor::THPSizeNode::str)
      .def("unbind", &torch::nested_tensor::THPSizeNode::unbind)
      .def(
          "__eq__",
          [](torch::nested_tensor::THPSizeNode& a,
             torch::nested_tensor::THPSizeNode& b) {
            return a.get_size_node() == b.get_size_node();
          })
      .def("__repr__", &torch::nested_tensor::THPSizeNode::str)
      .def("__len__", &torch::nested_tensor::THPSizeNode::len);

  // NOTE: Never forget about pybind return value policies
  // since you can expect transparent changes to the constiuents
  // via unbind.
  py::class_<torch::nested_tensor::THPNestedTensor>(m, "NestedTensor")
      .def_property_readonly(
          "dtype", &torch::nested_tensor::THPNestedTensor::getDtype)
      .def_property_readonly(
          "layout", &torch::nested_tensor::THPNestedTensor::getLayout)
      .def_property_readonly(
          "device", &torch::nested_tensor::THPNestedTensor::getDevice)
      .def_property_readonly(
          "requires_grad",
          &torch::nested_tensor::THPNestedTensor::requires_grad)
      .def("__len__", &torch::nested_tensor::THPNestedTensor::len)
      .def("element_size", &torch::nested_tensor::THPNestedTensor::element_size)
      .def(
          "nested_size",
          py::overload_cast<>(
              &torch::nested_tensor::THPNestedTensor::nested_size))
      .def(
          "nested_size",
          py::overload_cast<int64_t>(
              &torch::nested_tensor::THPNestedTensor::nested_size))
      .def(
          "nested_stride",
          &torch::nested_tensor::THPNestedTensor::nested_stride)
      .def(
          "unbind",
          [](torch::nested_tensor::THPNestedTensor self) {
            std::vector<py::object> result;
            // FOR BUFFER
            if (self.data().is_right()) {
              if (self.nested_dim() == 1) {
                for (int64_t i = 0; i < self.len(); i++) {
                  result.push_back(torch::jit::toPyObject(
                      self.data().right().get_structure().payload(i)));
                }
              } else {
                std::vector<int64_t> split_sizes;
                for (int64_t i = 0; i < self.len(); i++) {
                  split_sizes.push_back(size_node_memory(
                      self.data().right().nested_size().children(i),
                      self.data().right().nested_stride().children(i)));
                }
                std::vector<at::Tensor> buffers = at::split_with_sizes(
                    self.data().right().get_buffer(),
                    c10::IntArrayRef(split_sizes),
                    0);
                for (int64_t i = 0; i < self.len(); i++) {
                  result.push_back(
                      py::cast(torch::nested_tensor::THPNestedTensor(
                          torch::nested_tensor::_BufferNestedTensor(
                              buffers[i],
                              self.data().right().nested_size().children(i),
                              self.data().right().nested_stride().children(
                                  i)))));
                }
              }
              return result;
            }

            // FOR LIST
            if (self.nested_dim() == 1) {
              for (int64_t i = 0; i < self.len(); i++) {
                result.push_back(torch::jit::toPyObject(
                    self.data().left().get_structure().payload(i)));
              }
            } else {
              for (int64_t i = 0; i < self.len(); i++) {
                result.push_back(py::cast(torch::nested_tensor::THPNestedTensor(
                    torch::nested_tensor::_ListNestedTensor(
                        self.data().left().get_structure().children(i)))));
              }
            }
            return result;
          })
      .def("size", &torch::nested_tensor::THPNestedTensor::size)
      .def(
          "requires_grad_",
          &torch::nested_tensor::THPNestedTensor::requires_grad_)
      .def("numel", &torch::nested_tensor::THPNestedTensor::numel)
      .def_property_readonly(
          "grad", &torch::nested_tensor::THPNestedTensor::grad)
      .def("detach", &torch::nested_tensor::THPNestedTensor::detach)
      .def("dim", &torch::nested_tensor::THPNestedTensor::dim)
      .def("pin_memory", &torch::nested_tensor::THPNestedTensor::pin_memory)
      .def("nested_dim", &torch::nested_tensor::THPNestedTensor::nested_dim)
      .def("is_pinned", &torch::nested_tensor::THPNestedTensor::is_pinned)
      .def(
          "is_contiguous",
          &torch::nested_tensor::THPNestedTensor::is_contiguous)
      .def("contiguous", &torch::nested_tensor::THPNestedTensor::contiguous)
      .def("get_buffer", &torch::nested_tensor::THPNestedTensor::get_buffer)
      .def("to_tensor", &torch::nested_tensor::THPNestedTensor::to_tensor)
      .def("__str__", &torch::nested_tensor::THPNestedTensor::str)
      .def("__repr__", &torch::nested_tensor::THPNestedTensor::str);

  // NOTE: This is a private function until it is feature complete
  m.def("_jit_tensorwise", &torch::nested_tensor::jit_tensorwise);
  m.def("as_nested_tensor", &torch::nested_tensor::as_nested_tensor);
}
