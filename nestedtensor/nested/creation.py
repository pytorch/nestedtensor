import torch
import numbers

from . import nested
from . import utils
from nestedtensor import _C

def as_nested_tensor(data, dtype=None, device=None):
    # Simple wrapper around a nested list of Tensors.
    # Shares memory with original objects.
    # # TODO: Needs tests to check failure cases
    ret_impl = _C.as_nested_tensor(data)
    ret = nested.NestedTensor(ret_impl)
    if dtype is not None:
        ret = ret.to(dtype)
    if device is not None:
        ret = ret.to(device)
    return ret

def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Arguments match torch.tensor
    """
    if utils.is_nested_tensor(data):
        # This is consistent with torch.tensor(torch.Tensor)
        # but errors out.
        raise ValueError("To copy construct from a NestedTensor, "
                         "use sourceTensor.clone().detach() or "
                         "sourceTensor.clone().detach().requires_grad_(True), "
                         "rather than torch.tensor(sourceTensor).")
    elif torch.is_tensor(data):
        # The user has the right to expect a NestedTensor from this
        # function, but we can't meaningfully provide one if passed a Tensor
        raise ValueError("Can't construct a NestedTensor from a Tensor")
    else:
        result = as_nested_tensor(data).contiguous()

        if dtype is not None or device is not None:
            result = result.to(dtype=dtype, device=device)
        if requires_grad:
            result = result.requires_grad_(requires_grad)
        if pin_memory:
            result = result.pin_memory()
        return result
