import torch
import numbers
import warnings

from . import nested
from nestedtensor import _C


def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Arguments match torch.tensor
    """
    if requires_grad:
        warnings.warn(
            "NestedTensor temporarily does not support autograd. Please use an older commit.")

    result = nested.NestedTensor(_C.nested_tensor_impl(data))

    if dtype is not None or device is not None:
        result = result.to(dtype=dtype, device=device)
    if requires_grad:
        result = result.requires_grad_(requires_grad)
    if pin_memory:
        result = result.pin_memory()
    return result


def as_nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if requires_grad:
        warnings.warn(
            "NestedTensor temporarily does not support autograd. Please use an older commit.")
    # TODO: Needs tests to check failure cases
    if not isinstance(data, nested.NestedTensor):
        data = nested_tensor(data, dtype, device, requires_grad, pin_memory)
    if not(dtype is None and device is None and requires_grad is None and pin_memory is None):
        if dtype is not None or device is not None:
            data = data.to(dtype=dtype, device=device)
        if requires_grad:
            data = data.requires_grad_(requires_grad)
        if pin_memory:
            data = data.pin_memory()
    return data
