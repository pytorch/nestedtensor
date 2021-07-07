import torch
import numbers
import warnings

from . import nested
import nestedtensor


def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False, channels_last=False):
    """
    Arguments match torch.tensor
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if channels_last is None:
        channels_last = False
    return nested.NestedTensor(nestedtensor._C.nested_tensor_impl(data, dtype, device, requires_grad, pin_memory, channels_last))


def as_nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
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
