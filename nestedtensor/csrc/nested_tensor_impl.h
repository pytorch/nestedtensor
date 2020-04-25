#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutogradTensorId;

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

  IntArrayRef sizes() const override;

  torch::nested_tensor::NestedTensor _data;

};

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor._data.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

}
