#include <creation.h>
#include <python_nested_tensor.h>

namespace torch {
namespace nested_tensor {

THPNestedTensor cos_out(const THPNestedTensor& input) {
  const TensorNode data_node = input.get_structure();
  return THPNestedTensor(make_contiguous(
      map([](at::Tensor data) { return data.cos(); }, data_node)));
}

THPNestedTensor cos_out(const THPNestedTensor& input, THPNestedTensor out) {
  THPNestedTensor result = cos_out(input);
  TensorNode result_node = result.get_structure();
  TensorNode out_node = out.get_structure();
  apply(
      [](Tensor& out, Tensor& result) { out.copy_(result); },
      out_node,
      result_node);
  return out;
}

THPNestedTensor cos(const THPNestedTensor& self) {
  return cos_out(self);
}

THPNestedTensor cos_(THPNestedTensor& self) {
  return cos_out(self, self);
}

} // namespace nested_tensor
} // namespace torch
