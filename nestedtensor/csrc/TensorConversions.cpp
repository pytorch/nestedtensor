#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

namespace at {

using namespace torch::nested_tensor;
using namespace c10;

Tensor NestedTensor_to_dtype_layout(
    const Tensor& self,
    const TensorOptions& options,
    bool non_blocking,
    bool copy,
    c10::optional<MemoryFormat> memory_format) {
  return autograd_map_nested_tensor(
      [&](at::Tensor t) {
        return at::native::to(t, options, non_blocking, copy, memory_format);
      },
      self);
}
// Tensor NestedTensor_to_device(
//     const Tensor& self,
//     Device device,
//     ScalarType dtype,
//     bool non_blocking,
//     bool copy,
//     c10::optional<MemoryFormat> memory_format) {
// }
// Tensor NestedTensor_to_dtype(
//     const Tensor& self,
//     ScalarType dtype,
//     bool non_blocking,
//     bool copy,
//     c10::optional<MemoryFormat> memory_format) {
//   return autograd_map_nested_tensor(
//       [&](at::Tensor t) {
//         return at::native::to(t, dtype, non_blocking, copy, memory_format);
//       },
//       self);
// }
// Tensor NestedTensor_to_other(
//     const Tensor& self,
//     const Tensor& other,
//     bool non_blocking,
//     bool copy,
//     c10::optional<MemoryFormat> memory_format) {
//   return autograd_map_nested_tensor(
//       [&](at::Tensor s, at::Tensor o) {
//         return at::native::to(s, o, non_blocking, copy, memory_format);
//       },
//       self,
//       other);
// }

// static auto registry =
//     torch::RegisterOperators()
//         .op("nestedtensor::to_dtype_layout",
//             [](const Tensor& self,
//                const TensorOptions& options,
//                bool non_blocking,
//                bool copy,
//                c10::optional<MemoryFormat> memory_format) {
//               return NestedTensor_to_dtype_layout(
//                   self, options, non_blocking, copy, memory_format);
//             });
        // .op("nestedtensor::to_device", NestedTensor_to_device)
        // .op("nestedtensor::to_dtype", NestedTensor_to_dtype)
        // .op("nestedtensor::to_other", NestedTensor_to_other);
//        .op("nestedtensor::to_tensor",
//            [](Tensor tensor, c10::optional<int64_t> dim) {
//              return NestedTensor_to_tensor(tensor, dim);
//            })
//        .op("nestedtensor::to_nested_tensor",
//            [](Tensor tensor, c10::optional<int64_t> dim) {
//              return NestedTensor_to_nested_tensor(tensor, dim);
//            });

} // namespace at
