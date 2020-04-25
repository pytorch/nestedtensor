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

Tensor NestedTensor_contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  auto self_impl = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
  auto nt = self_impl->_data.contiguous();
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

bool NestedTensor_is_pinned(const Tensor& self) {
  auto self_impl = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
  std::cout << "DDD" << std::endl;
  return self_impl->_data.is_pinned();
}

std::vector<at::Tensor> NestedTensor_unbind(const at::Tensor &self, int64_t dim) {
  auto self_impl = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
  auto _data = self_impl->_data;
  dim = at::maybe_wrap_dim(dim, _data.dim());
  auto node = _data.get_structure();
  auto nested_dim = _data.nested_dim();
  if (nested_dim == 1) {
    if (dim == 0) {
      std::vector<at::Tensor> result;
      for (const auto& child : node.unbind()) {
        result.push_back(child.payload());
      }
      return result;
    } else {
      int64_t dim_max_size = 0;
      for (const auto& child : node.unbind()) {
        int64_t dim_size = child.payload().size(dim - 1);
        dim_max_size = dim_max_size > dim_size ? dim_max_size : dim_size;
      }
      std::vector<at::Tensor> result;
      result.resize(dim_max_size);
      for (const auto& child : node.unbind()) {
        std::vector<TensorNode> tensor_nodes;
        for (at::Tensor tensor : at::unbind(child.payload(), dim - 1)) {
          tensor_nodes.push_back(TensorNode(std::move(tensor)));
        }
        result.push_back(at::detail::make_tensor<NestedTensorImpl>(
            NestedTensor(std::move(tensor_nodes))));
      }
      return result;
    }
  }
  std::vector<at::Tensor> unbound_thp;
  for (auto child : node.unbind()) {
    unbound_thp.push_back(at::detail::make_tensor<NestedTensorImpl>(NestedTensor(std::move(child))));
  }
  if (dim == 0) {
    return unbound_thp;
  }
  std::vector<std::vector<TensorNode>> unbound;
  for (size_t i = 0; i < unbound_thp.size(); i++) {
    std::vector<at::Tensor> tmp = unbound_thp[i].unbind(dim - 1);
    for (size_t j = 0; j < tmp.size(); j++) {
      if (unbound.size() >= j) {
        unbound.resize(j + 1);
      }
      unbound[j].push_back(TensorNode(std::move(tmp[j])));
    }
  }
  std::vector<at::Tensor> result;
  for (size_t i = 0; i < unbound.size(); i++) {
    result.push_back(at::detail::make_tensor<NestedTensorImpl>(
        NestedTensor(TensorNode(std::move(unbound[i])))));
  }
  return result;
}

//TODO: CONTINUE HERE!
Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    if (self.has_names() && self.names()[dim] != Dimname::wildcard()) {
      TORCH_CHECK_INDEX(
          false,
          "select(): index ",
          index,
          " out of range for tensor of size ",
          self.sizes(),
          " at dimension ",
          self.names()[dim]);
    }
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    return select_sparse(self, dim, index);
  }
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  auto result = self.as_strided(sizes, strides, storage_offset);
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(
                        const Tensor&,
                        const Tensor&,
                        const Tensor&,
                        IntArrayRef,
                        IntArrayRef,
                        IntArrayRef,
                        int64_t),
                    &NestedTensor_conv2d>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor&, MemoryFormat memory_format),
                    &NestedTensor_contiguous>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema("aten::is_pinned(Tensor self) -> bool")
                .impl_unboxedOnlyKernel<
                    bool(const Tensor&),
                    &NestedTensor_is_pinned>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema("aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")
                .impl_unboxedOnlyKernel<
                    std::vector<Tensor>(const Tensor&, int64_t),
                    &NestedTensor_unbind>(NestedTensorKey));

;
}
