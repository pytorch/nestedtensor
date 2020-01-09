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
        if not (isinstance(data, list) or isinstance(data, tuple)):
            raise ValueError(
                "Pass a list or tuple to construct a NestedTensor. Got {} instead.".format(type(data)))

        def _type_check(data):
            num_nested_tensors = sum(
                [utils.is_nested_tensor(data_) for data_ in data])
            if num_nested_tensors > 0 and num_nested_tensors != len(data):
                raise ValueError(
                    "If an entry is a NestedTensor all other entries must be too.")
            num_tensors = sum([torch.is_tensor(data_) for data_ in data])
            if num_tensors > 0 and num_tensors != len(data):
                raise ValueError(
                    "If an entry is a Tensor all other entries must be too.")
            num_numbers = sum([isinstance(data_, numbers.Number)
                               for data_ in data])
            if num_numbers > 0 and num_numbers != len(data):
                raise ValueError(
                    "If an entry is a number all other entries must be too.")
            num_lists = sum([isinstance(data_, list) for data_ in data])
            if num_lists > 0 and num_lists != len(data):
                raise ValueError(
                    "If an entry is a list all other entries must be too.")
            num_tuples = sum([isinstance(data_, tuple) for data_ in data])
            if num_tuples > 0 and num_tuples != len(data):
                raise ValueError(
                    "If an entry is a tuple all other entries must be too.")

        _type_check(data)
        # print('data')
        # print(data)
        result = nested.NestedTensor(_C.nested_tensor(data))

        if dtype is not None or device is not None:
            result = result.to(dtype=dtype, device=device)
        if requires_grad:
            result = result.requires_grad_(requires_grad)
        if pin_memory:
            result = result.pin_memory()
        return result
