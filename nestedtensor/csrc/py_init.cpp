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

using namespace torch::nested_tensor;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<torch::nested_tensor::THPIValueNode>(m, "NestedList")
      .def("__str__", &THPIValueNode::str)
      .def("unbind", &THPIValueNode::unbind)
      .def("__repr__", &THPIValueNode::str)
      .def("__len__", &THPIValueNode::len);

  py::class_<torch::nested_tensor::THPSizeNode>(m, "SizeNode")
      .def("__str__", &THPSizeNode::str)
      .def("unbind", &THPSizeNode::unbind)
      .def("__repr__", &THPSizeNode::str)
      .def("__len__", &THPSizeNode::len)
      .def("__eq__", [](THPSizeNode& a_, THPSizeNode& b_) {
        SizeNode a = a_.get_node();
        SizeNode b = b_.get_node();
        if (a.height() != b.height()) {
          return false;
        }
        NestedNode<bool> tmp = torch::nested_tensor::map(
            [](c10::List<int64_t> a, c10::List<int64_t> b) {
              if (a.size() != b.size()) {
                return false;
              }
              for (size_t i = 0; i < a.size(); i++) {
                if (a[i] != b[i]) {
                  return false;
                }
              }
              return true;
            },
            a,
            b);
        auto fn = [](bool a, bool b) { return a && b; };
        return reduce<decltype(fn), bool, bool>(tmp, fn, true);
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
      .def("nested_size", &THPNestedTensor::nested_size)
      .def("nested_stride", &THPNestedTensor::nested_stride)
      .def(
          "unbind",
          [](THPNestedTensor self) {
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
                  result.push_back(py::cast(
                      THPNestedTensor(torch::nested_tensor::_BufferNestedTensor(
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
                result.push_back(py::cast(
                    THPNestedTensor(torch::nested_tensor::_ListNestedTensor(
                        std::move(child)))));
              }
              return py::cast(result);
            }
          })
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
      .def("__str__", &THPNestedTensor::str)
      .def("__repr__", &THPNestedTensor::str);

  // NOTE: This is a private function until it is feature complete
  m.def("_jit_tensorwise", &torch::nested_tensor::jit_tensorwise);
  m.def("as_nested_tensor", &torch::nested_tensor::as_nested_tensor);
  m.def("as_nested_list", &torch::nested_tensor::as_nested_list);
}
