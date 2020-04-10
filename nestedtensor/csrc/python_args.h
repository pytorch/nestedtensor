#include <torch/extension.h>

template <typename T>
struct IAR {
  std::vector<T> val;
  bool not_list;
  template <int64_t repeat>
  at::ArrayRef<T> extract() {
    if (not_list && val.size() < repeat) {
      for (T i = val.size(); i < repeat; i++) {
        val.push_back(val[0]);
      }
    }
    return at::ArrayRef<T>(val);
  }
};

namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<IAR<T>> {
 public:
  /**
   * This macro establishes the name 'inty' in
   * function signatures and declares a local variable
   * 'value' of type inty
   */
  PYBIND11_TYPE_CASTER(IAR<T>, _("int[]"));

  /**
   * Conversion part 1 (Python->C++).
   */
  bool load(handle obj, bool) {
    /* Extract PyObject from handle */
    if (py::isinstance<py::int_>(obj)) {
      value.val.push_back(py::cast<int64_t>(obj));
      value.not_list = true;
      return true;
    }
    if (py::isinstance<py::float_>(obj)) {
      value.val.push_back(py::cast<double>(obj));
      value.not_list = true;
      return true;
    }
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      value.val = py::cast<std::vector<T>>(obj);
      value.not_list = false;
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
      IAR<T> src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(src.val.data());
  }
};
} // namespace detail
} // namespace pybind11
