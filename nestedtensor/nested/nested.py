import torch

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

class NestedTensor(object):
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
                return getattr(self._impl, name)(*args, **kwargs)
            return _wrapped_fn
        return self.__dict__[name]

    # --- magic methods ---

    def __hash__(self):
        return hash(self._impl)

    def __eq__(self, other):
        if isinstance(other, NestedTensor):
            return _wrap_result(self._impl.__eq__(other._impl))
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
        raise NotImplementedError(
            "NestedTensor.to is currently not implemented.")
        return nestedtensor.as_nested_tensor(new_tensors)

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
        # TODO:This was disabled for now to focus on DETR
        if func is torch.nn.functional.linear:
            return _wrap_result(_nn_functional_linear(*impl_args, **impl_kwargs))
        if func is torch.nn.functional.embedding_bag:
            return _wrap_result(_nn_functional_embedding_bag(*impl_args, **impl_kwargs))
        if func is torch.nn.functional.batch_norm:
            return _wrap_result(_nn_functional_batch_norm(*impl_args, **impl_kwargs))
        if func is torch.nn.functional.adaptive_avg_pool2d:
            return _wrap_result(_nn_functional_adaptive_avg_pool2d(*impl_args, **impl_kwargs))
        if func is torch.nn.functional.multi_head_attention_forward:
            return _wrap_result(nestedtensor.nn.mha.multi_head_attention_forward(*args, **kwargs))
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
        tensor, mask = masking.to_tensor_mask(self, mask_dim)
        return tensor.masked_fill(~mask, padding)
