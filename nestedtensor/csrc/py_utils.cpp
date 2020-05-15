#include <nestedtensor/csrc/py_utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;

c10::optional<c10::IValue> py_obj_to_ivalue(py::object py_obj) {
    // std::cout << "c0 py_obj: " << py_obj << std::endl;
  auto inferred_type = tryToInferType(py_obj);
    // std::cout << "c1" << std::endl;
  if (!inferred_type.success()) {
    // std::cout << "c2" << std::endl;
    return c10::nullopt;
  }
    // std::cout << "c3" << std::endl;
  auto payload = toIValue(py_obj, inferred_type.type());
    // std::cout << "c4" << std::endl;
  return payload;
}

} // namespace nested_tensor
} // namespace torch
