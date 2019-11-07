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

from torch import _ListNestedTensor


DEBUG = int(os.getenv("DEBUG", 1))


def _first_tensor(self):
    if len(self) == 0:
        return None
    if self.nested_dim() == 1:
        return self.unbind()[0]
    else:
        return _first_tensor(self.unbind()[0])

def add_to_list_nested_tensor(f):
    setattr(_ListNestedTensor, f.__name__, f)
    return f

@add_to_list_nested_tensor
def is_contiguous(self):
    return False

@add_to_list_nested_tensor
def contiguous(self):
    return creation.nested_tensor(self.unbind())

@add_to_list_nested_tensor
def __str__(self):
    result = "nestedtensor([\n"
    for tensor in self.unbind():
        result += "  " + tensor.__str__() + ",\n"
    result += "])"
    return result

@add_to_list_nested_tensor
def __repr__(self):
    result = "nestedtensor([\n"
    for tensor in self.unbind():
        result += "  " + tensor.__repr__() + ",\n"
    result += "])"
    return result

@add_to_list_nested_tensor
def to_tensor(self):
    if self.nested_dim() == 1:
        return torch.stack(self.unbind())
    else:
        return torch.stack(list(map(lambda x: x.to_tensor(), self.unbind())))

@add_to_list_nested_tensor
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

@add_to_list_nested_tensor
def to(self, *args, **kwargs):
    return _ListNestedTensor([t.to(*args, **kwargs) for t in self.unbind()])

@add_to_list_nested_tensor
def numel(self):
    return sum(t.numel() for t in self.unbind())

@add_to_list_nested_tensor
def pin_memory(self):
    (t.pin_memory() for t in self.unbind())
