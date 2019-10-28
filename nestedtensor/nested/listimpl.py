import torch
import torch.nn.functional as F
import numbers
from functools import wraps
from . import codegen

from . import masking
from . import creation
from . import nested
from . import utils
import collections
import os


DEBUG = int(os.getenv("DEBUG", 1))


def _first_tensor(self):
    if len(self) == 0:
        return None
    if self.nested_dim() == 1:
        return self.unbind()[0]
    else:
        return _first_tensor(self.unbind()[0])


class _ListNestedTensor(object):
    def __init__(self, tensors):
        # Tuple disables changes in size via append etc.
        # assert isinstance(tensors, tuple)
        self._tensors = tensors
        if len(tensors) > 0:
            self._default_tensor = _first_tensor(self)
            if DEBUG:
                utils._verify_tensors(tensors)
        else:
            self._default_tensor = torch.rand(1)[0]

    def dim(self):
        return self._default_tensor.dim() + self.nested_dim()

    def is_pinned(self):
        return self._default_tensor.is_pinned()

    @property
    def dtype(self):
        return self._default_tensor.dtype

    @property
    def layout(self):
        return self._default_tensor.layout

    @property
    def device(self):
        return self._default_tensor.device

    @property
    def requires_grad(self):
        return self._default_tensor.requires_grad

    @property
    def grad(self):
        def _gather_grad_tensors(self):
            if self.nested_dim() == 1:
                return list(t.grad for t in self.unbind())
            else:
                return list(_gather_grad_tensors(t) for t in self.unbind())
        return _ListNestedTensor(_gather_grad_tensors(self))

    def requires_grad_(self, requires_grad=True):
        def _gather_requires_grad__tensors(self):
            if self.nested_dim() == 1:
                return list(t.requires_grad_(requires_grad) for t in self.unbind())
            else:
                return list(_gather_requires_grad__tensors(t) for t in self.unbind())
        return _ListNestedTensor(_gather_requires_grad__tensors(self))

    def detach(self):
        def _gather_detach_tensors(self):
            if self.nested_dim() == 1:
                return list(t.detach for t in self.unbind())
            else:
                return list(_gather_detach_tensors(t) for t in self.unbind())
        return _ListNestedTensor(_gather_detach_tensors(self))

    def backward(self, gradient, retain_graph, create_graph):
        for t, g in zip(self.unbind(), gradient.unbind()):
            t.backward(g, retain_graph, create_graph)

    # Cannot be decorated as _nested_property since
    # it's used for dispatch within the function
    def nested_dim(self):
        if len(self._tensors) == 0 or torch.is_tensor(self._tensors[0]):
            return 1
        else:
            return (self._tensors[0]).nested_dim() + 1

    def __len__(self):
        return len(self.unbind())

    def element_size(self):
        if DEBUG:
            utils._verify_tensors(self)
        return self._default_tensor.element_size()

    def unbind(self):
        # TODO: For now this is sometimes a list sometimes a tuple
        return self._tensors

    def is_contiguous(self):
        return False

    def contiguous(self):
        return creation.nested_tensor(self.unbind())

    def __str__(self):
        result = "nestedtensor([\n"
        for tensor in self._tensors:
            result += "  " + tensor.__str__() + ",\n"
        result += "])"
        return result

    def __repr__(self):
        result = "nestedtensor([\n"
        for tensor in self._tensors:
            result += "  " + tensor.__repr__() + ",\n"
        result += "])"
        return result

    def nested_size(self):
        if self.nested_dim() == 1:
            return tuple(t.size() for t in self._tensors)
        else:
            return tuple(t.nested_size() for t in self.unbind())

    def nested_stride(self):
        if self.nested_dim() == 1:
            return tuple(t.stride() for t in self._tensors)
        else:
            return tuple(t.nested_stride() for t in self.unbind())

    def to_tensor(self):
        if self.nested_dim() == 1:
            return torch.stack(self.unbind())
        else:
            return torch.stack(list(map(lambda x: x.to_tensor(), self.unbind())))

    def size(self, dim):
        if dim is not None:
            return self.size()[dim]
        all_sizes = tuple(t.size() for t in self.unbind())

        def compare_sizes(size, other_size):
            result_size = list(size)
            for i in range(len(size)):
                result_size[i] = size[i] if size[i] == other_size[i] else None
            return tuple(result_size)

        result_size = list(all_sizes[0])
        for size in all_sizes:
            result_size = compare_sizes(result_size, size)
        return (len(self),) + result_size

    def to(self, *args, **kwargs):
        return _ListNestedTensor([t.to(*args, **kwargs) for t in self.unbind()])

    def numel(self):
        return sum(t.numel() for t in self.unbind())

    def pin_memory(self):
        (t.pin_memory() for t in self.unbind())
