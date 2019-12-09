import torch
import torch.nn.functional as F
import numbers
from functools import wraps
from . import utils
import collections
import os
import nestedtensor

DEBUG = int(os.getenv("DEBUG", 0))


def _nested_numel(tup):
    if isinstance(tup, torch.Size):
        p = 1
        for t in tup:
            p *= t
        return p
    else:
        return sum(_nested_numel(t) for t in tup)


def _prod(tup):
    p = 1
    for t in tup:
        p *= t
    return p


def _infer_nested_dim(nested_size):
    if isinstance(nested_size, torch.Size):
        return 0
    # NOTE: This is 1 beause nested_size and therefore size is an empty tuple of length 0
    # This is consistent with the behavior of torch.tensor([])
    if len(nested_size) == 0:
        return 1
    return _infer_nested_dim(nested_size[0]) + 1


def _infer_dim(nested_size):
    if isinstance(nested_size, torch.Size):
        return len(nested_size)
    else:
        # NOTE: This is 1 beause nested_size and therefore size is an empty tuple of length 0
        # This is consistent with the behavior of torch.tensor([])
        if len(nested_size) == 0:
            return 1
        return _infer_dim(nested_size[0]) + 1


def _cont_strides(nested_size):
    if isinstance(nested_size, torch.Size):
        stride = (1,)
        for s in nested_size[:-1]:
            stride = (stride[-1] * s,) + stride
        return stride
    else:
        return tuple(map(_cont_strides, nested_size))


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
        self._buffer = nestedtensor._C._BufferNestedTensor(buffer_)
        # Tuple disables changes in size via append etc.
        # assert isinstance(tensors, tuple)
        if DEBUG:
            assert buffer_.dim() == 1
        self._nested_size = nested_size
        # Lazily initialized if None
        self._nested_stride = nested_stride
        self._nested_dim = _infer_nested_dim(self._nested_size)
        if len(self) > 0:
            self._meta_tensor = buffer_
        else:
            self._meta_tensor = torch.rand(1)[0]
        self._element_size = self._meta_tensor.element_size()
        self._dim = _infer_dim(self._nested_size)
        # self._is_pinned = _meta_tensor.is_pinned() NOTE: Expensive op!
        self._dtype = self._meta_tensor.dtype
        # self._layout = self._meta_tensor.layout NOTE: Can't pickle torch.layout object
        self._device = self._meta_tensor.device
        self._requires_grad = self._meta_tensor.requires_grad
        self._is_contiguous = True
        # Used to cache unbind
        self._unbound_tensors = None

    def get_buffer(self):
        return self._buffer.get_buffer()

    def dim(self):
        return self._dim

    def is_pinned(self):
        if len(self) > 0:
            return self.get_buffer().is_pinned()
        else:
            return False

    @property
    def dtype(self):
        return self._dtype

    @property
    def layout(self):
        # NOTE: Can't pickle torch.layout object
        return self._meta_tensor.layout

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad(self):
        return _BufferNestedTensor(self.get_buffer().grad,
                                   self.nested_size(), self.nested_stride())

    def requires_grad_(self, requires_grad=True):
        self.get_buffer().requires_grad_(requires_grad)
        return self

    def detach(self):
        return nested.NestedTensor(
            _BufferNestedTensor(self.get_buffer().detach,
                                self.nested_size(), self.nested_stride()))

    def backward(self, gradient, retain_graph, create_graph):
        for t, g in zip(self.unbind(), gradient.unbind()):
            t.backward(g, retain_graph, create_graph)

    def nested_dim(self):
        return self._nested_dim

    def __len__(self):
        return len(self._nested_size)

    def element_size(self):
        if DEBUG:
            utils._verify_tensors(self)
        return self._element_size

    def unbind(self):
        if self._unbound_tensors is not None:
            return self._unbound_tensors
        nested_size = self._nested_size
        if self.nested_dim() == 1:
            result = ()
            offset = 0
            for i in range(len(nested_size)):
                size = nested_size[i]
                tensor = self.get_buffer().narrow(
                    0, offset, _prod(size)).reshape(size)
                offset += tensor.numel()
                result = result + (tensor,)
        else:
            nested_stride = self.nested_stride()
            result = ()
            offset = 0
            for i in range(len(self)):
                sub_numel = _nested_numel(nested_size[i])
                result_i = _BufferNestedTensor(self.get_buffer().narrow(
                    0, offset, sub_numel), nested_size[i], nested_stride[i])
                offset += sub_numel
                result = result + (result_i,)
        self._unbound_tensors = result
        return self._unbound_tensors

    def is_contiguous(self):
        return self._is_contiguous

    def nested_size(self):
        return self._nested_size

    def nested_stride(self):
        if self._nested_stride is None:
            self._nested_stride = _cont_strides(self.nested_size())
        return self._nested_stride

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
            len_sizes = len(nested_size)
            if isinstance(nested_size[0], torch.Size):
                sizes = nested_size
            else:
                sizes = iter(_size(x) for x in nested_size)

            result = tuple(k[0] if k[1:] == k[:-1]
                           else None for k in zip(*sizes))
            return (len_sizes,) + result

        return _size(self.nested_size())

    def to(self, *args, **kwargs):
        return torch.NestedTensor(_BufferNestedTensor(self.get_buffer().to(*args, **kwargs),
                                                      self.nested_size(), self.nested_stride()))

    def numel(self):
        return self.get_buffer().numel()

    def pin_memory(self):
        self.get_buffer().pin_memory()

    def __str__(self):
        def _str(x, indent=0):
            if x.nested_dim() == 0:
                return ""
            s = indent*"\t" + "[\n"
            if x.nested_dim() == 1:
                strs = list(xi.__str__() for xi in x.unbind())
                strs = list(map(lambda xi: "\n".join(
                    map(lambda xij: (indent + 1)*"\t" + xij, xi.split("\n"))), strs))
                s += ",\n".join(strs)
            else:
                s += ",\n".join(list(map(
                    lambda xi: _str(xi, indent + 1), x.unbind())))
            s += "\n" + indent * "\t" + "]"
            return s
        return "nested_tensor(" + _str(self) + ")"
