#include <reductions.h>

namespace torch {
namespace nested_tensor {

template <class F>
void add_reduction(auto m, auto c, std::string name, F& at_out) {
}

void add_reductions_functions(
    pybind11::module,
    pybind11::class_<torch::nested_tensor::THPNestedTensor>) {

    add_reduction(m, c, "sum", at::sum);

}

} // namespace nested_tensor
} // namespace torch
