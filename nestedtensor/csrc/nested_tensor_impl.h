#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutograd;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(torch::nested_tensor::NestedTensor&& data)
      : TensorImpl(
            c10::DispatchKeySet(NestedTensorKey),
            data.dtype(),
            data.device()),
        _data(std::move(data)) {}

  int64_t dim() const override {
    return _data.dim();
  }
  int64_t numel() const override {
    return _data.numel();
  }
  bool is_contiguous(
      at::MemoryFormat memory_format) const override {
    return _data.is_contiguous();
  }

  std::vector<int64_t> sizes();
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

  torch::nested_tensor::NestedTensor _data;

};

inline bool is_nested_tensor_impl(const at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

inline at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline torch::nested_tensor::NestedTensor get_nested_tensor(
    const at::Tensor tensor) {
  return get_nested_tensor_impl(tensor)->_data;
}

inline torch::nested_tensor::TensorNode get_nested_tensor_structure(
    const at::Tensor tensor) {
  return get_nested_tensor(tensor).get_structure();
}

inline bool is_tensor_shape(const at::Tensor tensor) {
  auto nt = get_nested_tensor(tensor);
  for (const auto& size : nt.sizes()) {
    if (!size) {
      return false;
    }
  }
  return true;
}

inline at::Tensor wrap_nested_tensor(
    torch::nested_tensor::NestedTensor&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(std::move(result));
}

inline at::Tensor wrap_tensor_node(
    torch::nested_tensor::TensorNode&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(result)));
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_);

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor._data.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

}
