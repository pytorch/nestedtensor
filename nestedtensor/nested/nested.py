import torch
from .nested_python import NestedTensorPythonImpl
from .nested_c import NestedTensorCImpl
from . import masking
from torch._C import _disabled_torch_function_impl


def _wrap_result(result):
    if isinstance(result, list):
        return list(_wrap_result(r) for r in result)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    if isinstance(result, NestedTensorPythonImpl):
        return NestedTensor(result)
    if isinstance(result, NestedTensorCImpl):
        return NestedTensor(result)
    return result


def _filter_impl(args, kwargs):
    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        if isinstance(a, NestedTensor):
            impl_args.append(a._impl)
        elif torch.is_tensor(a):
            impl_args.append(a)
        elif isinstance(a, list):
            a_impl, _ = _filter_impl(a, {})
            impl_args.append(a_impl)
        elif isinstance(a, tuple):
            a_impl, _ = _filter_impl(a, {})
            impl_args.append(tuple(a_impl))
        else:
            impl_args.append(a)
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


class NestedTensor(metaclass=NestedTensorMeta):
    __torch_function__ = _disabled_torch_function_impl
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
        self._impl = impl

    def __getattr__(self, name):
        if hasattr(self._impl, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(self._impl, name)(*impl_args, **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return self.__dict__[name]

    # --- magic methods ---

    def __hash__(self):
        return hash(self._impl)

    def __eq__(self, other):
        return _wrap_result(self._impl.__eq__(other))

    def __ne__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl.__ne__(other._impl))
        return _wrap_result(self._impl.__ne__(other))

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

    @property
    def data(self):
        return _wrap_result(self._impl.data)

    @property
    def is_sparse(self):
        return self._impl.is_sparse

    def requires_grad_(self, requires_grad=True):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return _wrap_result(self._impl.requires_grad_(requires_grad))

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        impl = None
        if gradient is not None:
            if torch.is_tensor(gradient):
                impl = gradient
            else:
                impl = gradient._impl
        self._impl.backward(impl, retain_graph, create_graph)

    def nested_dim(self):
        """
        The nested dimension of ```self``` NestedTensor.
        The nested dimension is defined as the level of indexing required
        to reach a Tensor constiuent.
        """
        return self._impl.nested_dim()

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
        return len(self._impl)

    def size(self, dim=None):
        return self._impl.size(dim)

    def unbind(self, dim=0):
        return _wrap_result(self._impl.unbind(dim))

    def to(self, *args, **kwargs):
        raise NotImplementedError(
            "NestedTensor.to is currently not implemented.")

    def __str__(self):
        def _str(x, indent=0, tab="  "):
            if x.nested_dim() == 0:
                return ""
            s = indent*tab + "[\n"
            if x.nested_dim() == 1:
                strs = list(map(str, x.unbind()))
                strs = list(map(lambda xi: "\n".join(
                    map(lambda xij: (indent + 1)*tab + xij, xi.split("\n"))), strs))
                s += ",\n".join(strs)
            else:
                s += ",\n".join(list(map(
                    lambda xi: _str(xi, indent + 1), x.unbind())))
            s += "\n" + indent * tab + "]"
            return s
        return "nested_tensor(" + _str(self) + ")"

    def __repr__(self):
        return str(self)

    # --- impl forward ends ---

    # --- dependent on impl ---

    def to_tensor(self, dim=0):
        """
        Not necessarily a view.
        """
        return _wrap_result(self._impl.to_tensor(dim))

    def nested_size(self, dim=None):
        if dim is None:
            return self._impl.nested_size()
        return self._impl.nested_size(dim)

    def nested_stride(self, dim=None):
        return self._impl.nested_stride(dim)

    # --- dependent on impl ends ---

    def __torch_function__(self, func, types, args=(), kwargs=None):
        impl_args, impl_kwargs = _filter_impl(args, kwargs)
        return _wrap_result(func(*impl_args, **impl_kwargs))

    # Might require nonzero
    def __bool__(self):
        raise NotImplementedError(
            "NestedTensor doesn't support function __bool__")

    def __getitem__(self, key):
        return _wrap_result(self._impl.__getitem__(key))

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
        tensor, mask = masking.to_tensor_mask(self, mask_dim)
        return tensor.masked_fill(~mask, padding)
