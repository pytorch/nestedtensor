#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>

namespace at {

using namespace torch::nested_tensor;

// NOTE: Can't reuse dispatch from cos_ to cos_out either, because it requries
// support for empty through unary_op_impl
Tensor& NestedTensor_cos_(Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  auto f = [](at::Tensor tensor) { tensor.cos_(); };
  apply<decltype(f)>(std::move(f), self_impl->_data.get_structure());
  return self;
}

Tensor NestedTensor_cos(const Tensor& self) {
  auto self_impl = get_nested_tensor_impl(self);
  return at::detail::make_tensor<NestedTensorImpl>(
      map([](at::Tensor tensor) { return tensor.cos(); },
          self_impl->_data.get_structure()));
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::cos_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    Tensor&(Tensor& self),
                    &NestedTensor_cos_>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema("aten::cos(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor& self),
                    &NestedTensor_cos>(NestedTensorKey))
    ;

}
