import torch
import numbers

from . import nested
from nestedtensor import _C

def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Arguments match torch.tensor
    """
    result = nested.NestedTensor(_C.nested_tensor(data))

    if dtype is not None or device is not None:
        result = result.to(dtype=dtype, device=device)
    if requires_grad:
        result = result.requires_grad_(requires_grad)
    if pin_memory:
        result = result.pin_memory()
    return result


def as_nested_tensor(data, dtype=None, device=None):
    # TODO: Needs tests to check failure cases
    if not utils.is_nested_tensor(data):
        data = nested_tensor(data, dtype, device)
    if dtype is not None or device is not None:
        return data.to(dtype=dtype, device=device)
    return data
