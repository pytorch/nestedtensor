
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;

Tensor NestedTensor_zeros_like(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> memory_format) {
  std::vector<int64_t> numel_array;
  numel_array.push_back(self.numel());
  at::Tensor buffer =
      at::zeros(IntArrayRef(numel_array), dtype, layout, device, pin_memory);
  return wrap_buffer(std::move(buffer), get_nested_size(self));
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "zeros_like", NestedTensor_zeros_like);
}

} // namespace at
