
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

at::Tensor _to_tensor(TensorNode node) {
  // TODO: Recursive stacking is expensive.
  if (node.is_leaf()) {
    return node.payload();
  }
  if (node.degree() == 0) {
    return at::empty({0});
  }
  std::vector<at::Tensor> flat;
  for (auto child : node.unbind()) {
    flat.push_back(_to_tensor(child));
  }
  return stack(flat);
}

at::Tensor to_tensor(NestedTensorImpl* nt_impl) {
  // TODO: Not necessarily a view because of stack and reshape.
  std::vector<int64_t> new_size;
  for (const auto& si : nt_impl->opt_sizes()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t at
      // this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  return _to_tensor(nt_impl->get_structure());
}

struct NestedTensorFunction_to_tensor
    : public torch::autograd::Function<NestedTensorFunction_to_tensor> {
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input) {
    // TODO: Not necessarily a view because of stack and reshape.
    std::vector<int64_t> new_size;
    auto impl_data = get_nested_tensor_impl(input);
    for (const auto& si : impl_data->opt_sizes()) {
      if (!si) {
        // TODO: This assumes we'll extend to_tensor to also work with int64_t
        // at this level.
        throw std::out_of_range(
            "to_tensor()/to_tensor(0) only works if there is no None in size().");
      }
      new_size.push_back(*si);
    }
    ctx->save_for_backward({input});
    return _to_tensor(impl_data->get_structure());
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(grad_output_.size() == 1, "grad_output must be of size 1.");
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor grad_output = grad_output_[0];
    return {wrap_tensor_node(torch::nested_tensor::impl::build_structure(
        std::move(grad_output.clone().reshape({-1})),
        get_nested_tensor_impl(input)->nested_size()))};
  }
};

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_) {
  if (!dim_) {
    return NestedTensorFunction_to_tensor::apply(tensor);
  }
  int64_t dim = maybe_wrap_dim((*dim_), tensor.dim());
  if (dim == 0) {
    return NestedTensorFunction_to_tensor::apply(tensor);
  }
  TORCH_CHECK(
      false, "Non-zero dimension ", *dim_, " is currently not supported.");
  // // If dim is bigger than nested_dim the NestedTensor is already
  // // of Tensor for dimensions bigger than the given.
  // if (impl_data->nested_dim() == 1) {
  //   return tensor;
  // }
  // // At this point nested_dim is at least 2. That means any unbind
  // // operation of a child must yield NestedTensors.
  // // If dim is 1 then we'll apply to_tensor(0) to the children and must
  // expect
  // // Tensors.
  // std::vector<at::Tensor> unbound = at::unbind(tensor, 0);
  // std::vector<TensorNode> result;
  // for (Tensor child : unbound) {
  //   auto ci = NestedTensor_to_tensor(child, dim - 1);
  //   if (is_nested_tensor_impl(ci)) {
  //     auto s = get_nested_tensor_impl(ci)->get_structure();
  //     result.push_back(TensorNode(std::move(s)));
  //   } else {
  //     // TODO: If it's a NestedTensor instance get the structure
  //     result.push_back(TensorNode(std::move(ci)));
  //   }
  // }
  // return wrap_tensor_node(TensorNode(std::move(result)));
}

TensorNode _unbind_tensors(TensorNode structure) {
  std::vector<TensorNode> result_nodes;
  if (structure.is_leaf()) {
    for (at::Tensor tensor : structure.payload().unbind()) {
      result_nodes.emplace_back(TensorNode(std::move(tensor)));
    }
  } else {
    for (TensorNode child : structure.unbind()) {
      result_nodes.emplace_back(_unbind_tensors(child));
    }
  }
  return TensorNode(std::move(result_nodes));
}

Tensor NestedTensor_to_nested_tensor(
    at::Tensor tensor,
    c10::optional<int64_t> dim_) {
  int64_t dim = 0;
  if (dim_) {
    dim = at::maybe_wrap_dim(*dim_, tensor.dim());
  }
  int64_t nested_dim = get_nested_tensor_impl(tensor)->nested_dim();
  // if dim < nested_dim() the NestedTensor is already nested
  // up to the given dimension.
  if (dim < nested_dim) {
    return tensor;
  }
  TensorNode unbound = _unbind_tensors(get_nested_tensor_structure(tensor));
  for (int64_t i = 0; i < (dim - nested_dim); i++) {
    unbound = _unbind_tensors(unbound);
  }
  return wrap_tensor_node(std::move(unbound));
}

static auto registry =
    torch::RegisterOperators()
        .op("nestedtensor::to_tensor",
            [](Tensor tensor, c10::optional<int64_t> dim) {
              return NestedTensor_to_tensor(tensor, dim);
            })
        .op("nestedtensor::to_nested_tensor",
            [](Tensor tensor, c10::optional<int64_t> dim) {
              return NestedTensor_to_nested_tensor(tensor, dim);
            });

} // namespace at
