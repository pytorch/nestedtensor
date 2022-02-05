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
        return nested_tensor(data, dtype, device, requires_grad, pin_memory)
    return data
