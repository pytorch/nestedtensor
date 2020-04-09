#include <torch/extension.h>

struct IAR {
  std::vector<int64_t> val;
  bool is_int;
  template <int64_t repeat>
  at::IntArrayRef extract() {
    if (is_int && val.size() < repeat) {
      for (int64_t i = val.size() ; i < repeat; i++) {
        val.push_back(val[0]);
      }
    }
    return at::IntArrayRef(val);
  }
};

namespace pybind11 {
namespace detail {
template <>
struct type_caster<IAR> {
 public:
  /**
   * This macro establishes the name 'inty' in
   * function signatures and declares a local variable
   * 'value' of type inty
   */
  PYBIND11_TYPE_CASTER(IAR, _("int[]"));

  /**
   * Conversion part 1 (Python->C++).
   */
  bool load(handle obj, bool) {
    /* Extract PyObject from handle */
    if (py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj)) {
      value.val.push_back(py::cast<int64_t>(obj));
      value.is_int = true;
      return true;
    }
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      value.val = py::cast<std::vector<int64_t>>(obj);
      value.is_int = false;
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
      IAR src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(src.val.data());
  }
};
} // namespace detail
} // namespace pybind11
