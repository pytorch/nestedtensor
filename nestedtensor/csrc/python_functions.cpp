#include <python_functions.h>
#include <functions.h>

namespace torch {
namespace nested_tensor {

void add_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
          auto copy_fn = [](THPNestedTensor self, THPNestedTensor source,
              bool non_blocking=false) {
            self.data().copy_(source.data());
          }
  m.def("copy_",copy_fn
  );
  c.def("copy_", copy_fn);
}

}
}
