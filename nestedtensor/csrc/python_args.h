#include <torch/extension.h>

namespace torch {
namespace python_args {

template <class T, class F>
c10::optional<T> optional_lift(F& fn, c10::optional<py::object> obj) {
  if (!obj) {
    return c10::nullopt;
  }
  return fn(*obj);
}

at::Scalar to_scalar(py::object);

} // namespace python_args
} // namespace torch
