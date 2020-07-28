import torch
import numbers
from . import masking

from . import creation

import nestedtensor


def _new_torch_stack(tensors, dim=0, out=None):
    result = torch.ops.nestedtensor.stack(list(
        t._impl if isinstance(t, NestedTensor) else t for t in tensors), dim)
    result = _wrap_result(result)
    if out is None:
        return result
    out.copy_(result)

def _new_torch_cat(tensors, dim=0, out=None):
    result = torch.ops.nestedtensor.cat(list(
        t._impl if isinstance(t, NestedTensor) else t for t in tensors), dim)
    result = _wrap_result(result)
    if out is None:
        return result
    out.copy_(result)

def _wrap_result(result):
    if isinstance(result, list):
        return list(_wrap_result(r) for r in result)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    return (
        NestedTensor(result)
        if torch.is_tensor(result) and torch.ops.nestedtensor.is_nested_tensor_impl(result)
        else result
    )


def _filter_impl(args, kwargs):
    if kwargs is None:
        kwargs = {}
    impl_args = [a._impl if isinstance(a, NestedTensor) else a for a in args]
    impl_kwargs = {
        k: v._impl if isinstance(v, NestedTensor) else v for (k, v) in kwargs.items()
    }
    return impl_args, impl_kwargs


class NestedTensorMeta(type):
    def __getattr__(cls, name):
        if getattr(torch.Tensor, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(impl_args[0], name)(
                    *(impl_args[1:]), **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return self.__dict__[name]

# -------------------------NestedTensor core---------------------------


class NestedTensor(metaclass=NestedTensorMeta):
    # The attributes must match across all constiuents
    #
    # The NestedTensor's attributes then become that of its
    # constiuents.
    #
    # data must be a list of Tensors or NestedTensors
    #
    # Attributes:
    #     dim()
    #     layout
    #     device
    #     dtype
    #     requires_grad
    #     is_pinned()
    # Neighbors may share data, maybe all share data.
    # Levels of contiguity
    def __init__(self, impl):
        if not torch.ops.nestedtensor.is_nested_tensor_impl(impl):
            raise TypeError("Got unexpected type " + str(type(impl)))
        self._impl = impl

    def __getattr__(self, name):
        if getattr(self._impl, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(self._impl, name)(*impl_args, **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return self.__dict__[name]

    # --- magic methods ---

    def __eq__(self, other):
        return _wrap_result(self._impl.__eq__(other._impl))

    def __ne__(self, other):
        return _wrap_result(self._impl.__ne__(other._impl))

    def __add__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl + other._impl)
        return _wrap_result(self._impl + other)

    def __radd__(self, other):
        assert not isinstance(other, NestedTensor)
        return _wrap_result(self._impl + other)

    def __mul__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl * other._impl)
        return _wrap_result(self._impl * other)

    def __rmul__(self, other):
        assert not isinstance(other, NestedTensor)
        return _wrap_result(self._impl * other)

    def __sub__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl - other._impl)
        return _wrap_result(self._impl - other)

    def __rsub__(self, other):
        assert not isinstance(other, NestedTensor)
        return _wrap_result(other - self._impl)

    def __truediv__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl / other._impl)
        return _wrap_result(self._impl / other)

    def __floordiv__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl // other._impl)
        return _wrap_result(self._impl // other)

    def __pow__(self, *args, **kwargs):
        impl_args, impl_kwargs = _filter_impl(args, kwargs)
        return _wrap_result(self._impl.__pow__(*impl_args, **impl_kwargs))

    def __rpow__(self, exponent):
        assert not isinstance(exponent, NestedTensor)
        return _wrap_result(torch.pow(exponent, self._impl))

    @property
    def shape(self):
        return self.size()

    @property
    def dtype(self):
        """
        The data type of ```self``` NestedTensor.
        """
        return self._impl.dtype

    @property
    def layout(self):
        """
        The layout of ```self``` NestedTensor.
        """
        return self._impl.layout

    @property
    def device(self):
        """
        The device of ```self``` NestedTensor.
        """
        return self._impl.device

    @property
    def requires_grad(self):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return self._impl.requires_grad

    @property
    def grad(self):
        """
        This attribute is None by default and becomes a NestedTensor the
        first time a call to backward() computes gradients for self.
        The attribute will then contain the gradients computed and future
        calls to backward() will accumulate (add) gradients into it.
        """
        return _wrap_result(self._impl.grad)

    def requires_grad_(self, requires_grad=True):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return _wrap_result(self._impl.requires_grad_(requires_grad))

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        self._impl.backward(gradient._impl, retain_graph, create_graph)

    def nested_dim(self):
        """
        The nested dimension of ```self``` NestedTensor.
        The nested dimension is defined as the level of indexing required
        to reach a Tensor constiuent.
        """
        return torch.ops.nestedtensor.nested_dim(self._impl)

    def tensor_dim(self):
        """
        The tensor dimension of ```self``` NestedTensor.
        The tensor dimension is defined as the dimension of the Tensor constiuents.
        """
        return self.dim() - self.nested_dim()

    def __len__(self):
        """
        The number of entries in the list ```self``` represents.
        """
        return torch.ops.nestedtensor.len(self._impl)

    def size(self, dim=None):
        if dim is not None:
            return self.size()[dim]
        return tuple(torch.ops.nestedtensor.sizes(self._impl))

    def to(self, *args, **kwargs):
        # TODO: to is currently not supported by impls due to argparsing.
        new_tensors = [t.to(*args, **kwargs) for t in self.unbind()]
        # TODO: Make contiguous by default? Heavy operation...
        # NOTE: Needs grad support, which nestedtensor.nested_tensor
        # constructor doesn't have.
        return nestedtensor.as_nested_tensor(new_tensors)

    def __str__(self):
        return torch.ops.nestedtensor.str(self._impl)

    def __repr__(self):
        return torch.ops.nestedtensor.str(self._impl)

    # --- impl forward ends ---

    # --- dependent on impl ---

    def to_tensor(self, dim=0):
        """
        Not necessarily a view.
        """
        return _wrap_result(torch.ops.nestedtensor.to_tensor(self._impl, dim))

    def __repr__(self):
        # TODO: This relies on the fact that repr is not implemented compliant with
        # the purpose of repr for torch.Tensor. Therefore returning str is ok.
        return self.__str__()

    def nested_size(self, dim=None):
        return nestedtensor._C.nested_size(self._impl, dim)

    def nested_stride(self, dim=None):
        return nestedtensor._C.nested_stride(self._impl, dim)

    # --- dependent on impl ends ---

    def __torch_function__(self, func, types, args=(), kwargs=None):
        impl_args, impl_kwargs = _filter_impl(args, kwargs)
        # Need a specialized implementation to support lists of lists of sizes.
        if func is torch.nn.functional.interpolate:
            return _wrap_result(nestedtensor._C.interpolate(*impl_args, **impl_kwargs))
        # Need a specialized implementation to dodge call to view in nll_loss
        if func is torch.nn.functional.cross_entropy:
            return _wrap_result(
                nestedtensor._C.cross_entropy(*impl_args, **impl_kwargs)
            )
        return _wrap_result(func(*impl_args, **impl_kwargs))

    # Might require nonzero
    def __bool__(self):
        raise NotImplementedError(
            "NestedTensor doesn't support function __bool__")

    def __getitem__(self, key):
         return _wrap_result(nestedtensor._C.get_item(self._impl, key))

    def __iter__(self):
        return iter(self.unbind())

    def to_nested_tensor(self, dim=0):
        return _wrap_result(torch.ops.nestedtensor.to_nested_tensor(self._impl, dim))

    def to_list(self):
        return self._impl.to_list()

    def to_tuple(self):
        return self._impl.to_tuple()

    def to_tensor_mask(self, mask_dim=None):
        """Returns a named tuple TensorMask with two tensors (tensor, mask)
        of dim equal to self.dim(). Tensor will contain all data of NestedTensor,
        expect that each tensor constiuent has been padded with 0s to equal the
        largest Tensor.

        The mask is a bool tensor with a 1-to-1 correspondence to each
        element of tensor. If an entry is True, the corresponding element
        stores data that is represented by self, if it is False it is a padding
        element. These two tensors can be used to contruct a NestedTensor, however,
        nested_dim will be lost in this process."""

        return masking.to_tensor_mask(self, mask_dim)

    def to_padded_tensor(self, mask_dim=None, padding=-1):
        tensor, mask = masking.to_tensor_mask(self.to_list(), mask_dim)
        return tensor.masked_fill(~mask, padding)
