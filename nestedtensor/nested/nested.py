import torch
import numbers
from . import masking

from . import creation

import nestedtensor
import warnings


def _not_impl_raise(cond, msg):
    if (isinstance(cond, bool) and cond) or (not isinstance(cond, bool) and cond is not None):
        raise NotImplementedError(
            msg + " is not supported yet. Please file an issue on https://github.com/pytorch/nestedtensor")


def _nn_functional_linear(input, weight, bias=None):
    # TODO: This is done because autograd/engine.cpp has an is_expandable_to check
    # that doesn't support NT's extension of the .sizes() function. Therefore
    # we need to disable the addition of NTs and Ts below autograd, but we still need
    # it for linear (hence add lives above autograd). Also linear insists on using the
    # in-place version, for which we don't have an op above autograd, since the custom
    # function wrapper autograd_map_nested_tensor doesn't support it.
    # And that's why we're writing our own version of linear here.
    output = input.matmul(weight.t())
    if bias is not None:
        output = output + bias
    return output


def _nn_functional_batch_norm(input, running_mean, running_var, weight=None, bias=None,
                              training=False, momentum=0.1, eps=1e-5):
    return torch.batch_norm(
        input, weight, bias, running_mean, running_var,
        training, momentum, eps, torch.backends.cudnn.enabled
    )


def _nn_functional_adaptive_avg_pool2d(input, output_size):
    return torch._C._nn.adaptive_avg_pool2d(input, output_size)


def _nn_functional_embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                                 scale_grad_by_freq=False, mode='mean', sparse=False,
                                 per_sample_weights=None, include_last_offset=False,
                                 padding_idx=None):
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == torch.long and input.is_floating_point():
        warnings.warn("Argument order of nn.functional.embedding_bag was changed. "
                      "Usage `embedding_bag(weight, input, ...)` is deprecated, "
                      "and should now be `embedding_bag(input, weight, ...)`.")
        weight, input = input, weight

    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError("embedding_bag: If per_sample_weights ({}) is not None, "
                         "then it must have the same shape as the input ({})"
                         .format(per_sample_weights.shape, input.shape))

    _not_impl_raise(max_norm, "max_norm")
    _not_impl_raise(per_sample_weights, "per_sample_weights")

    input_dim = torch.ops.nestedtensor.get_dim(input)
    if input_dim == 2:
        if offsets is not None:
            type_str = "<unknown>"
            # TODO: Remove this once script supports type() calls
            if not torch.jit.is_scripting():
                type_str = str(type(offsets))
            raise ValueError("if input is 2D, then offsets has to be None"
                             ", as input is treated is a mini-batch of"
                             " fixed length sequences. However, found "
                             "offsets of type {}".format(type_str))
        offsets_ = NestedTensor(input).nested_size()
        offsets = torch.zeros(len(offsets_), dtype=torch.int64)
        for i in range(1, len(offsets)):
            offsets[i] = offsets[i - 1] + offsets_[i - 1][0]
        offsets = offsets.to(input.device)
    elif input_dim == 1:
        raise ValueError("input has to be 2D NestedTensor,"
                         " but got NestedTensor of dimension {}".format(input_dim))
    if mode == 'sum':
        mode_enum = 0
    elif mode == 'mean':
        mode_enum = 1
    elif mode == 'max':
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError(
                "max mode does not support scaling the gradient by the frequency")

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if per_sample_weights is not None and mode != 'sum':
        raise NotImplementedError("embedding_bag: per_sample_weights was not None. "
                                  "per_sample_weights is only supported for mode='sum' "
                                  "(got mode='{}'). Please open a feature request on GitHub."
                                  .format(mode))
    if padding_idx is not None:
        raise NotImplementedError(
            "padding_idx is not supported for NestedTensor embedding_bag")

    ret, _, _, _ = torch.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode_enum,
        sparse,
        per_sample_weights,
        include_last_offset)
    return ret


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


def sum_to_size(tensor, shape):
    impl_args, _ = _filter_impl([tensor, shape], {})
    return _wrap_result(nestedtensor._C.sum_to_size(*impl_args))


def sizes_equal(tensor, shape):
    impl_args, _ = _filter_impl([tensor, shape], {})
    return _wrap_result(nestedtensor._C.sizes_equal(*impl_args))


def native_is_expandable_to(tensor, shape):
    impl_args, _ = _filter_impl([tensor, shape], {})
    return _wrap_result(nestedtensor._C.native_is_expandable_to(*impl_args))


def to_nested_tensor(tensor, dim=0):
    return _wrap_result(
        torch.ops.nestedtensor.to_nested_tensor(tensor._impl if isinstance(tensor, NestedTensor) else tensor, dim))


def transpose_nchw_nhwc(tensor):
    return _wrap_result(
        torch.ops.nestedtensor.transpose_nchw_nhwc(tensor._impl))


def transpose_nhwc_nchw(tensor):
    return _wrap_result(
        torch.ops.nestedtensor.transpose_nhwc_nchw(tensor._impl))


class NestedTensorMeta(type):
    def __getattr__(cls, name):
        if getattr(torch.Tensor, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(impl_args[0], name)(
                    *(impl_args[1:]), **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return cls.__dict__[name]

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

    def numel(self):
        return torch.ops.nestedtensor.get_numel(self._impl)

    def dim(self):
        return torch.ops.nestedtensor.get_dim(self._impl)

    def contiguous(self):
        if self.is_contiguous():
            return self
        return _wrap_result(torch.ops.nestedtensor.make_contiguous(self._impl))

    def is_contiguous(self, memory_format=torch.contiguous_format):
        if (memory_format == torch.contiguous_format):
            return torch.ops.nestedtensor.get_is_contiguous(self._impl, 0)
        if (memory_format == torch.channels_last):
            return torch.ops.nestedtensor.get_is_contiguous(self._impl, 2)
        raise RuntimeError("Given memory format " + str(memory_format) + " not supported.")

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
        return _wrap_result(self._impl.to(*args, **kwargs))

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
            return _wrap_result(nestedtensor.nn.multi_head_attention_forward(*args, **kwargs))
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

    def to_tensor_list(self):
        return torch.ops.nestedtensor.to_tensor_list(self._impl)

    def to_packed_sequence(self):
        if not self.dim() == 3 and self.nested_dim() == 1:
            raise RuntimeError(
                "NestedTensor should consistent of 2d Tensors of size L x *")
        return torch.nn.utils.rnn.pack_sequence(self.to_tensor_list(), enforce_sorted=False)

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

        # Return a tuple of a tensor and a mask that represent the given tensor list
        # Returned tensor is always the same no matter what mask_dim was passed.
        # If mask_dim was not passed, a mask with the smallest dimensionality would be returned.
        # if passed mask_dim is lower than the minimal dimensionality of the mask that can represent
        # the data tensor, an error is thrown.
        return torch.ops.nestedtensor.to_tensor_mask(self, mask_dim)

    def to_padded_tensor(self, padding=-1):
        padding = float(padding)
        return torch.ops.nestedtensor.to_padded_tensor(self, padding)

    def to_sparse_csr_tensor(self):
        return torch.ops.nestedtensor.to_sparse_csr(self._impl)
