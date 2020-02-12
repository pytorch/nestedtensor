#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

THPNestedTensor cos(
    const THPNestedTensor& input,
    c10::optional<THPNestedTensor> out) {
  TensorNode data_node = input.get_structure();
  TensorNode result_node =
      map([](const at::Tensor& data) { return data.cos(); }, data_node);
  if (out) {
      TensorNode out_node = out.get_structure();
      apply([](Tensor& out, const Tensor& result) {
              out.copy_(result);
              }, out_node, result_node);
      return *out;
  }
}

} // namespace nested_tensor
} // namespace torch
