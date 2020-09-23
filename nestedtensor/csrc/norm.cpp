#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

// Tensor NestedTensor_batch_norm(
//     const Tensor& input,
//     const c10::optional<Tensor>& weight,
//     const c10::optional<Tensor>& bias,
//     const c10::optional<Tensor>& running_mean,
//     const c10::optional<Tensor>& running_var,
//     bool training,
//     double momentum,
//     double eps,
//     bool cudnn_enabled) {
//   if (weight && bias) {
//     return autograd_map_nested_tensor(
//         [&](at::Tensor t, at::Tensor w, at::Tensor b) {
//           return at::batch_norm(
//                      t.unsqueeze(0),
//                      w,
//                      b,
//                      running_mean,
//                      running_var,
//                      training,
//                      momentum,
//                      eps,
//                      cudnn_enabled)
//               .squeeze(0);
//         },
//         input,
//         *weight,
//         *bias);
//   }
//   return autograd_map_nested_tensor(
//       [&](at::Tensor t) {
//         return at::batch_norm(
//                    t.unsqueeze(0),
//                    c10::nullopt,
//                    c10::nullopt,
//                    running_mean,
//                    running_var,
//                    training,
//                    momentum,
//                    eps,
//                    cudnn_enabled)
//             .squeeze(0);
//       },
//       input);
// }
// 
// TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
//   nt_impl(m, "batch_norm", NestedTensor_batch_norm);
// }

} // namespace at
