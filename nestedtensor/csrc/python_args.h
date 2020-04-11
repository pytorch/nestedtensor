#include <torch/extension.h>

template <typename T>
struct THPArrayRef {
  std::vector<T> val;
  bool is_list;
  template <int64_t repeat>
  at::ArrayRef<T> extract() {
    if (!is_list && val.size() == 1) {
      for (int64_t i = 1; i < repeat; i++) {
        val.push_back(val[0]);
      }
    }
    return at::ArrayRef<T>(val);
  }
};

namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<THPArrayRef<T>> {
 public:
  /**
   * This macro establishes the name 'inty' in
   * function signatures and declares a local variable
   * 'value' of type inty
   */
  PYBIND11_TYPE_CASTER(THPArrayRef<T>, _("int[]"));

  /**
   * Conversion part 1 (Python->C++).
   */
  bool load(handle obj, bool) {
    /* Extract PyObject from handle */
    if (py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj)) {
      value.val.push_back(py::cast<T>(obj));
      value.is_list = false;
      return true;
    }
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      value.val = py::cast<std::vector<T>>(obj);
      value.is_list = true;
      return true;
    }
    return false;
  }

  /**
   * Conversion part 2 (C++ -> Python) convert an instance into
   * a Python object. The second and third arguments are used to
   * indicate the return value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  static handle cast(
      THPArrayRef<T> src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(src.val.data());
  }
};
} // namespace detail
} // namespace pybind11
