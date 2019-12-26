#include <buffer_nested_tensor.h>
// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

enum impl_type {BUFFER, LIST};

using namespace torch::jit;
using namespace torch::autograd::utils;

struct THPNestedTensor {
  THPNestedTensor() = delete;
  THPNestedTensor(_BufferNestedTensor data);
  THPNestedTensor(_ListNestedTensor data);
  torch::autograd::Variable get_buffer() {
    return _data.get_buffer();
  }
  int64_t element_size() {
    return _data.element_size();
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
  bool requires_grad() {
    return _data.requires_grad();
  }
  _BufferNestedTensor data() {
    return _data;
  }
  py::object nested_size() {
    return wrap_nested_node(_data.nested_size());
  }
  py::object nested_stride() {
    return wrap_nested_node(_data.nested_stride());
  }
  THPNestedTensor requires_grad_(py::bool_ requires_grad) {
    return THPNestedTensor(_data.requires_grad_(requires_grad));
  }
  THPNestedTensor grad() {
    return THPNestedTensor(_data.grad());
  }
  THPNestedTensor detach() {
    return THPNestedTensor(_data.detach());
  }
  THPNestedTensor pin_memory() {
    return THPNestedTensor(_data.pin_memory());
  }
  std::string str() {
    return _NestedNode___str__(_data.get_structure());
  }
  int64_t len() {
    return _data.__len__();
  }
  bool is_pinned() {
    return _data.is_pinned();
  }
  int64_t nested_dim() {
    return _data.nested_dim();
  }
  int64_t dim() {
    return _data.dim();
  }
  int64_t numel() {
    return _data.numel();
  }
  at::Tensor to_tensor() {
    return _data.to_tensor();
  }
  bool is_contiguous() {
    return _data.is_contiguous();
  }

 private:
  _BufferNestedTensor _data;
  _impl_type
};

} // namespace nested_tensor
} // namespace torch
