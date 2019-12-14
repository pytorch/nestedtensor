import torch
import torch.nn.functional as F
import numbers
from functools import wraps
from . import utils
from . import nested
import collections
import os
import nestedtensor

DEBUG = int(os.getenv("DEBUG", 0))


def _prod(tup):
    p = 1
    for t in tup:
        p *= t
    return p


def _nested_numel(nested_size):
    if len(nested_size) == 0 or not isinstance(nested_size[0], list):
        return _prod(nested_size)
    else:
        return sum(_nested_numel(t) for t in nested_size)


def _nested_tensor_to_buffer(nested_tensor):
    """
    Given a nested tensor, return a new contiguous buffer covering all Tensor constiuents data.
    Returns a view if possible.
    """
    if nested_tensor.is_contiguous():
        return nested_tensor._buffer
    if nested_tensor.nested_dim() == 1:
        return torch.cat([t.flatten() for t in nested_tensor.unbind()], dim=0)
    else:
        return torch.cat([_nested_tensor_to_buffer(t) for t in nested_tensor], dim=0)


class _BufferNestedTensor(object):
    def __init__(self, buffer_, nested_size, nested_stride=None):
        # Tuple disables changes in size via append etc.
        # assert isinstance(tensors, tuple)
        if DEBUG:
            assert buffer_.dim() == 1
        nested_size=list(nested_size)
        if nested_stride is None:
            self._c_impl = nestedtensor._C._BufferNestedTensor(
                buffer_, nested_size)
        else:
            self._c_impl = nestedtensor._C._BufferNestedTensor(
                buffer_, nested_size, nested_stride)
        # self._nested_dim=_infer_nested_dim(nested_size)
        # self._dim=_infer_dim(nested_size)
        self._is_contiguous=True
        # Used to cache unbind
        self._unbound_tensors=None

    def get_buffer(self):
        return self._c_impl.get_buffer()

    def dim(self):
        return self._c_impl.dim()

    def is_pinned(self):
        if len(self) > 0:
            return self.get_buffer().is_pinned()
        else:
            return False

    @property
    def dtype(self):
        return self._c_impl.dtype

    @property
    def layout(self):
        # NOTE: Can't pickle torch.layout object
        return self._c_impl.layout

    @property
    def device(self):
        return self._c_impl.device

    @property
    def requires_grad(self):
        return self._c_impl.requires_grad

    @property
    def grad(self):
        return _BufferNestedTensor(self.get_buffer().grad,
                                   self.nested_size(), self.nested_stride())

    def requires_grad_(self, requires_grad=True):
        self.get_buffer().requires_grad_(requires_grad)
        return self

    def detach(self):
        return _BufferNestedTensor(self.get_buffer().detach,
                                self.nested_size(), self.nested_stride())

    def backward(self, gradient, retain_graph, create_graph):
        for t, g in zip(self.unbind(), gradient.unbind()):
            t.backward(g, retain_graph, create_graph)

    def nested_dim(self):
        return self._c_impl.nested_dim()

    def __len__(self):
        return len(self.nested_size())

    def element_size(self):
        if DEBUG:
            utils._verify_tensors(self)
        return self._c_impl.element_size()

    def unbind(self):
        return self._c_impl.unbind()

    def is_contiguous(self):
        return self._is_contiguous

    def nested_size(self):
        return self._c_impl.nested_size()

    def nested_stride(self):
        return self._c_impl.nested_stride()

    def to_tensor(self):
        """
        Not a view.
        """
        return self.get_buffer().reshape(self.size(None))

    def size(self, dim):
        # TODO: Unused until _ListNestedTensor has its own implementation
        if dim is not None:
            return self.size(None)[dim]

        def _size(nested_size):
            len_sizes=len(nested_size)
            if isinstance(nested_size[0], torch.Size):
                sizes=nested_size
            else:
                sizes=iter(_size(x) for x in nested_size)

            result=tuple(k[0] if k[1:] == k[:-1]
                           else None for k in zip(*sizes))
            return (len_sizes,) + result

        return _size(self.nested_size())

    def to(self, *args, **kwargs):
        return _BufferNestedTensor(self.get_buffer().to(*args, **kwargs),
                                                      self.nested_size(), self.nested_stride())

    def numel(self):
        return self.get_buffer().numel()

    def pin_memory(self):
        self.get_buffer().pin_memory()

    def __str__(self):
        def _str(x, indent=0):
            if x.nested_dim() == 0:
                return ""
            s=indent*"\t" + "[\n"
            if x.nested_dim() == 1:
                strs=list(xi.__str__() for xi in x.unbind())
                strs=list(map(lambda xi: "\n".join(
                    map(lambda xij: (indent + 1)*"\t" + xij, xi.split("\n"))), strs))
                s += ",\n".join(strs)
            else:
                s += ",\n".join(list(map(
                    lambda xi: _str(xi, indent + 1), x.unbind())))
            s += "\n" + indent * "\t" + "]"
            return s
        return "nested_tensor(" + _str(self) + ")"
