#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/functions.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace at {

using namespace torch::nested_tensor;

at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

// TODO: The dispatcher has trouble with this so we register an unboxed kernel.
Tensor NestedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  // auto nt = NestedTensor(at::ones({2, 3, 2, 1}));
  auto input_impl = static_cast<NestedTensorImpl*>(input.unsafeGetTensorImpl());
  auto nt = torch::nested_tensor::conv2d(
      input_impl->_data, weight, bias, stride, padding, dilation, groups);
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

// TODO: Can't have mixed return types in C++
// for different input values.
Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_) {
  auto impl_data = get_nested_tensor_impl(tensor)->_data;
  if (!dim_) {
  // std::cout << "LLL00" << std::endl;
    return impl_data.to_tensor();
  }
  int64_t dim = maybe_wrap_dim((*dim_), impl_data.dim());
  if (dim == 0) {
  // std::cout << "LLL11" << std::endl;
    return impl_data.to_tensor();
  }
  // If dim is bigger than nested_dim the NestedTensor is already
  // of Tensor for dimensions bigger than the given.
  if (impl_data.nested_dim() == 1) {
  // std::cout << "LLL22" << std::endl;
    return tensor;
  }
  // At this point nested_dim is at least 2. That means any unbind
  // operation of a child must yield NestedTensors.
  // If dim is 1 then we'll apply to_tensor(0) to the children and must expect
  // Tensors.
  // std::cout << "LLL" << std::endl;
  std::vector<at::Tensor> unbound = at::unbind(tensor, 0);
  std::vector<TensorNode> result;
  for (Tensor child : unbound) {
    auto ci = NestedTensor_to_tensor(child, dim - 1);
    if (ci.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
      auto ci_impl = static_cast<NestedTensorImpl*>(ci.unsafeGetTensorImpl());
      auto s = ci_impl->_data.get_structure();
      result.push_back(TensorNode(std::move(s)));
    } else {
      // std::cout << "ci: " << ci << std::endl;
      // TODO: If it's a NestedTensor instance get the structure
      result.push_back(TensorNode(std::move(ci)));
    }
  }
  return at::detail::make_tensor<at::NestedTensorImpl>(
      NestedTensor(TensorNode(std::move(result))));
}

bool NestedTensor_is_pinned(const Tensor& self) {
  auto self_impl = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
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
      std::vector<std::vector<TensorNode>> unbound;
      unbound.resize(dim_max_size);
      for (const auto& child : node.unbind()) {
        std::vector<at::Tensor> unbound_tensors =
            at::unbind(child.payload(), dim - 1);
        for (size_t i = 0; i < unbound_tensors.size(); i++) {
          unbound[i].push_back(TensorNode(std::move(unbound_tensors[i])));
        }
      }
      std::vector<at::Tensor> result;
      for (size_t i = 0; i < unbound.size(); i++) {
        TensorNode tmp = TensorNode(std::move(unbound[i]));
        result.push_back(at::detail::make_tensor<NestedTensorImpl>(NestedTensor(std::move(tmp))));
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
Tensor NestedTensor_select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  if (dim == 0) {
    TORCH_CHECK_INDEX(false, "select() only supports dim == 0 for now.");
  }
  auto input_impl = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
  TensorNode tn = input_impl->_data.get_structure().unbind()[index];
  torch::nested_tensor::NestedTensor nt = torch::nested_tensor::NestedTensor(
      std::move(tn));
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

// TODO: Could have a NestedTensorIterator that does binary ops
Tensor NestedTensor_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  TensorNode result_tensor_node =
      map([alpha](at::Tensor a, at::Tensor b) { return at::add(a, b, alpha); },
          self_impl->_data.get_structure(),
          other_impl->_data.get_structure());
  return at::detail::make_tensor<NestedTensorImpl>(
      NestedTensor(std::move(result_tensor_node)));
}

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

Tensor NestedTensor_all(const Tensor& self) {
  std::cout << "HEEE ALL " << std::endl;
  auto self_impl = get_nested_tensor_impl(self)->_data;
  if (self_impl.numel() == 0) {
    // XXX: self.options doesn't work here because
    // we don't want a Tensor backed by a NestedTensor
    Tensor result = at::empty({0}, at::kBool); //, self.options());
    result.fill_(1);
    return result;
  }
  auto map_all = flatten(map(
        [](at::Tensor tensor) { return tensor.all(); },
          self_impl.get_structure()));
  at::Tensor gathered = at::empty(
      {static_cast<int64_t>(map_all.size())}, at::kBool); //, self.options());
  for (size_t i = 0; i < map_all.size(); i++) {
    // std::cout << "map_all[" << i << "]: " << map_all[i] << std::endl;
    gathered[i] = map_all[i];
  }
  // std::cout << "gathered.all(): " << gathered.all() << std::endl;
  return gathered.all();
}

Tensor NestedTensor_any(const Tensor& self) {
  std::cout << "HEEE any " << std::endl;
  auto self_impl = get_nested_tensor_impl(self)->_data;
  if (self_impl.numel() == 0) {
    // XXX: self.options doesn't work here because
    // we don't want a Tensor backed by a NestedTensor
    Tensor result = at::empty({0}, at::kBool); //, self.options());
    result.fill_(1);
    return result;
  }
  auto map_any = flatten(map(
        [](at::Tensor tensor) { return tensor.any(); },
          self_impl.get_structure()));
  at::Tensor gathered = at::empty(
      {static_cast<int64_t>(map_any.size())}, at::kBool); //, self.options());
  for (size_t i = 0; i < map_any.size(); i++) {
    // std::cout << "map_any[" << i << "]: " << map_any[i] << std::endl;
    gathered[i] = map_any[i];
  }
  // std::cout << "gathered.any(): " << gathered.any() << std::endl;
  return gathered.any();
}

Tensor NestedTensor_eq(const Tensor& self, const Tensor& other) {
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  return at::detail::make_tensor<NestedTensorImpl>(
  map([](const Tensor a, const Tensor b) {
      auto c = at::eq(a, b);
      // std::cout << "c: " << c << std::endl;
      return c;
      }, 
      self_impl->_data.get_structure(),
      other_impl->_data.get_structure()));
}

Tensor NestedTensor_ne(const Tensor& self, const Tensor& other) {
  std::cout << "NE" << std::endl;
  auto self_impl = get_nested_tensor_impl(self);
  auto other_impl = get_nested_tensor_impl(other);
  return at::detail::make_tensor<NestedTensorImpl>(
  map([](const Tensor a, const Tensor b) {
      auto c = at::ne(a, b);
      // std::cout << "c: " << c << std::endl;
      return c;
      }, 
      self_impl->_data.get_structure(),
      other_impl->_data.get_structure()));
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
                    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor&, const Tensor&),
                    &NestedTensor_eq>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor&, const Tensor&),
                    &NestedTensor_ne>(NestedTensorKey))
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
                .schema(
                    "aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")
                .impl_unboxedOnlyKernel<
                    std::vector<Tensor>(const Tensor&, int64_t),
                    &NestedTensor_unbind>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(
                        const Tensor& self,
                        const Tensor& other,
                        Scalar alpha),
                    &NestedTensor_add>(NestedTensorKey))
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
        .op(torch::RegisterOperators::options()
                .schema("aten::all(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor& self),
                    &NestedTensor_all>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema("aten::any(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor& self),
                    &NestedTensor_any>(NestedTensorKey))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    Tensor(const Tensor&, int64_t, int64_t),
                    &NestedTensor_select>(NestedTensorKey));

;
}
