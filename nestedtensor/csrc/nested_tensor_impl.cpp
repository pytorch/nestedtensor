#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/functions.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace at {

using namespace torch::nested_tensor;

// TODO: The dispatcher has trouble with this so we register an unboxed kernel.
Tensor NestedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  // auto nt = NestedTensor(at::ones({2, 3, 2, 1}));
  auto input_impl = static_cast<NestedTensorImpl*>(input.unsafeGetTensorImpl());
  std::cout << "HERE : "  << *input_impl << std::endl;
  auto nt = torch::nested_tensor::conv2d(
      input_impl->_data, weight, bias, stride, padding, dilation, groups);
  std::cout << "MADE" << std::endl;
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

IntArrayRef NestedTensorImpl::sizes() const {
  std::vector<c10::optional<int64_t>> size = _data.sizes();
  std::vector<int64_t> sizes;
  for (auto opt_int : size) {
    if (opt_int) {
      sizes.push_back(*opt_int);
    } else {
      throw std::runtime_error("NestedTensor size is not Tensor shape compliant.");
    }
  }
  return IntArrayRef(sizes);
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &NestedTensor_conv2d>(NestedTensorKey))
  ;

}
