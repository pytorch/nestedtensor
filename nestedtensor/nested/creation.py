import torch
import numbers
import warnings


def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Arguments match torch.tensor
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    # if nestedtensor.USE_C_EXTENSION:
    #     from . import nested_c
    #     return nested.NestedTensor(
    #         nested_c.NestedTensorCImpl(_C.nested_tensor_impl(
    #             data, dtype, device, requires_grad, pin_memory)))
    from . import nested_python
    assert not pin_memory
    impl = nested_python.nested_tensor_python(data,
                                              dtype=dtype, device=device,
                                              requires_grad=requires_grad)
    from .nested import NestedTensor
    return NestedTensor(impl)


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
