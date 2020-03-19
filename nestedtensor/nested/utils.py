import torch
import torch.nn.functional as F
import numbers
from functools import wraps
import collections
import os

from .nested import NestedTensor
from . import creation

from collections.abc import Iterable
from itertools import repeat

from nestedtensor import _C

DEBUG = int(os.getenv("DEBUG", 1))

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _check_is_contiguous(self):
    if self.nested_dim() == 1:
        return all(t.is_contiguous() for t in self.unbind())
    else:
        return all(_check_is_contiguous(t) for t in self.unbind())

def _verify_tensors(tensors):
    if torch.is_nested_tensor(tensors):
        tensors = tensors.unbind()
    if not torch.is_tensor(tensors[0]):
        tensors = [_verify_tensors(tensor_list) for tensor_list in tensors]
    default_tensor = tensors[0]
    for tensor in tensors:
        assert torch.is_tensor(tensor)
    dim = default_tensor.dim()
    layout = default_tensor.layout
    device = default_tensor.device
    dtype = default_tensor.dtype
    requires_grad = default_tensor.requires_grad
    # TODO: Doesn't work in DataLoader
    # TODO: Make this DEBUG only
    # TODO: support is_pinned = default_tensor.is_pinned()
    for tensor in tensors:
        if not (dim == tensor.dim() and
                layout == tensor.layout and
                device == tensor.device and
                dtype == tensor.dtype and
                requires_grad == tensor.requires_grad):  # and
            #                is_pinned == tensor.is_pinned()):
            raise ValueError("Each passed Tensor "
                             "must match in dim, layout, "
                             "device, dtype and requires_grad")
    return default_tensor


def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


def find_nested_tensor_dispatch_key(*args):
    """
    Returns first instance of NestedTensor within arguments.
    The search continues depth first for list arguments.
    """
    for arg in args:
        if is_nested_tensor(arg):
            return arg
        if isinstance(arg, list):
            for a in arg:
                if is_nested_tensor(a):
                    return a
    return None


def _wrap_dim(self, dim):
    if isinstance(dim, tuple):
        return tuple(_wrap_dim(self, d) for d in dim)
    if dim is None:
        return None
    if dim < 0:
        dim = self.dim() + dim
    if dim >= self.dim():
        raise ValueError(
            "Index {} out of bounds for NestedTensor of dim {}".format(dim, self.dim()))
    return dim


def _gen_unbound(unb_args, dim_args, *args, **kwargs):
    # Unbind everything via __getitem__ that is either NestedTensor or in unbind_args
    # All args to-be-unbound should match in length

    dispatch_key = find_nested_tensor_dispatch_key(*args)
    key_len = len(dispatch_key)

    def _subtract_one(dim):
        if isinstance(dim, tuple):
            return tuple(_subtract_one(d) for d in dim)
        else:
            result = dim - 1
            if result < 0:
                raise ValueError(
                    "Dimension {} out of bounds or invalid dimension.".format(dim))
            return result

    unbound_args = []
    for i, arg in enumerate(args):
        if is_nested_tensor(arg) or i in unb_args:
            assert len(arg) == key_len
            unbound_args.append(tuple(arg[i] for i in range(key_len)))
        else:
            unbound_args.append(tuple(arg for _ in range(key_len)))
        if i in dim_args:
            unbound_args[i] = _subtract_one(unbound_args[i])

    unbound_kwargs = []
    for k, arg in kwargs.items():
        if is_nested_tensor(arg) or k in unb_args:
            assert len(arg) == key_len
            new_kwarg = tuple((k, arg[i]) for i in range(key_len))
        else:
            new_kwarg = tuple((k, arg) for _ in range(key_len))
        if k in dim_args:
            new_kwarg[1] = _subtract_one(new_kwarg[1])
        unbound_kwargs.append(new_kwarg)

    args_gen = zip(*unbound_args)
    if len(unbound_kwargs) == 0:
        for new_args in args_gen:
            yield (new_args, {})
    else:
        for new_args, new_kwargs in zip(args_gen, zip(*unbound_kwargs)):
            yield (new_args, dict(new_kwargs))


def _unwrap_tensor_tuples(l):
    if torch.is_tensor(l):
        return (l,)
    if len(l) > 0 and isinstance(l[0], tuple):
        return tuple(zip(*l))
    return tuple(zip(*[_unwrap_tensor_tuples(li) for li in l]))


def match_type_signature_prefix(types, args):
    for t, a in zip(types, args):
        if not isinstance(a, t):
            return False
    return True

# The assumption is that f can handle a list of tensors
# This is used to write tensor-wise functions
# The resulting function accepts a multiple NestedTensors as arguments
# and calls f tensor-wise
# Make nested_stride optional (cont. by default)
# Return flattened tensor pairs, then create _BufferNestedTensor impl directly
def tensorwise(unbind_args=None, dim_args=None, wrap_dim_args=True):

    if unbind_args is None:
        unbind_args = []
    if dim_args is None:
        dim_args = []

    def wrapper(f):
        @wraps(f)
        def decorator(*_args, **_kwargs):
            def _func(*args, **kwargs):
                if find_nested_tensor_dispatch_key(*args) is None:
                    result = f(*args, **kwargs)
                    if not torch.is_tensor(result):
                        return tuple(result)
                    return result
                else:
                    results = []
                    for local_args, local_kwargs in _gen_unbound(unbind_args, dim_args, *args, **kwargs):
                        results.append(_func(*local_args, **local_kwargs))
                    return results
            dispatch_key = find_nested_tensor_dispatch_key(*_args)
            if dispatch_key is None:
                return f(*_args, **_kwargs)
            else:
                if wrap_dim_args:
                    args = []
                    for i, a in enumerate(_args):
                        if i in dim_args:
                            a = _wrap_dim(dispatch_key, a)
                        args.append(a)
                    kwargs = {}
                    for k, v in _kwargs.items():
                        if k in dim_args:
                            kwargs[k] = _wrap_dim(dispatch_key, v)
                        kwargs[k] = v
                else:
                    args = _args
                    kwargs = _kwargs
                results = _func(*args, **kwargs)
                results = _unwrap_tensor_tuples(results)
                if len(results) == 1:
                    return creation.nested_tensor(results[0])
                return tuple(map(creation.nested_tensor, results))

        return decorator
    return wrapper


def _get_tensortype_args(args, kwargs):
    # Includes NestedTensor
    tt_args = []
    for a in args:
        if is_nested_tensor(a) or torch.is_tensor(a):
            tt_args.append(a)
    for _, v in kwargs.items():
        if is_nested_tensor(v) or torch.is_tensor(v):
            tt_args.append(v)
    return tt_args


def _get_nestedtensor_args(args, kwargs):
    nt_args = []
    for a in args:
        if is_nested_tensor(a):
            nt_args.append(a)
    for _, v in kwargs.items():
        if is_nested_tensor(v):
            nt_args.append(v)
    return nt_args


def _arg_apply(fn, args, kwargs):
    new_args = []
    for i in range(len(args)):
        new_args.append(fn(args[i]))
    new_kwargs = {}
    for (k, v) in kwargs.items():
        new_kwargs[k] = fn(v)
    return new_args, new_kwargs


def reduction(support_nested_dim=True, unbind_args=None, dim_args=None):
    # associative, commutitative reductions that return Tensors
    # Use tensorwise
    if dim_args is None:
        dim_args = [1, 'dim']

    def wrapper(f):
        tf = tensorwise(unbind_args=unbind_args, dim_args=dim_args)(f)

        @wraps(tf)
        def decorator(self, *args, **kwargs):
            if len(args) > 0:
                dim = _wrap_dim(self, args[0])
                args = args[1:]
            else:
                dim = kwargs.pop('dim', None)
                dim = _wrap_dim(self, dim)
            if dim is None:
                if self.is_contiguous():
                    return f(self._impl.get_buffer(), *args, **kwargs)
                else:
                    raise ValueError("Not supported")
            elif isinstance(dim, tuple):
                result = self
                for d in sorted(list(dim))[::-1]:
                    result = decorator(result, d, *args, **kwargs)
                return result
            else:
                if dim > self.nested_dim() - 1:
                    return tf(self, dim, *args, **kwargs)
                else:
                    if not support_nested_dim:
                        raise ValueError(
                            "Reduction over given nested dimension {} not implemented.")
                    unbound = self.unbind(dim)
                    if is_nested_tensor(unbound[0]):
                        nested_size = unbound[0].nested_size()
                        if not all(nested_size == t.nested_size() for t in unbound):
                            for t in unbound:
                                if nested_size != t.nested_size():
                                    raise ValueError(
                                        ("Cannot reduce across dimension {}. "
                                         "Shapes {} and {} don't match.").format(
                                            dim, nested_size, t.nested_size()))
                    buffer_ = tf(torch.stack([t.contiguous()._impl.get_buffer()
                                              for t in unbound]), 0, *args, **kwargs)
                    if is_nested_tensor(unbound[0]):
                        return NestedTensor(_C._BufferNestedTensor(buffer_,
                                                                           nested_size,))
                    else:
                        return buffer_.reshape_as(unbound[0])
        return decorator
    return wrapper
