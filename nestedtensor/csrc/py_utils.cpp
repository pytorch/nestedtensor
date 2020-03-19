#include <py_utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;

c10::optional<c10::IValue> py_obj_to_ivalue(py::object py_obj) {
  auto inferred_type = tryToInferType(py_obj);
  if (!inferred_type.success()) {
    return c10::nullopt;
  }
  auto payload = toIValue(py_obj, inferred_type.type());
  return payload;
}

} // namespace nested_tensor
} // namespace torch
