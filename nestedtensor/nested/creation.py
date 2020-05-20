import torch
import numbers

from . import nested
from . import utils
from nestedtensor import _C

def as_nested_tensor(data, dtype=None, device=None):
    # Simple wrapper around a nested list of Tensors.
    # Shares memory with original objects.
    # # TODO: Needs tests to check failure cases
    if utils.is_nested_tensor(data):
        return data
    return nested_tensor(data, dtype, device)

def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Arguments match torch.tensor
    """
    result = as_nested_tensor(data).contiguous()

    if dtype is not None or device is not None:
        result = result.to(dtype=dtype, device=device)
    if requires_grad:
        result = result.requires_grad_(requires_grad)
    if pin_memory:
        result = result.pin_memory()
    return result
