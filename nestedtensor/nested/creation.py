import torch
import numbers

from . import nested
from . import utils
from nestedtensor import _C

def as_nested_tensor(data, dtype=None, device=None):
    # Simple wrapper around a nested list of Tensors.
    # Shares memory with original objects.
    # # TODO: Needs tests to check failure cases
    ret = nested.NestedTensor(_C._ListNestedTensor(data))
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

        def _to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, nested.NestedTensor):
                return data
            if isinstance(data, list) or isinstance(data, tuple):
                return list(_to_tensor(data_) for data_ in data)
            return torch.tensor(data)

        def _create_buffer(data):
            def __flatten_data(data):
                if isinstance(data, torch.Tensor):
                    return [data.flatten()]  # This data will be copied implicitly via cat
                elif isinstance(data, nested.NestedTensor):
                    return [data.contiguous()._impl.get_buffer()]
                else:
                    result = []
                    for data_i in data:
                        result += __flatten_data(data_i)
                    return result
            flat_data = __flatten_data(data)
            return torch.cat(flat_data)

        def _cont_stride(size):
            stride = (1,)
            for s in size[:0:-1]:
                stride = (stride[0] * s,) + stride
            return stride

        def _nested_size(data):
            if isinstance(data, torch.Tensor):
                return list(data.size())
            if isinstance(data, nested.NestedTensor):
                return data._impl.nested_size()
            return list(_nested_size(t) for t in data)

        def _nested_stride(data):
            if isinstance(data, torch.Tensor):
                return list(data.stride())
            if isinstance(data, nested.NestedTensor):
                return data._impl.nested_stride()
            return list(_nested_stride(t) for t in data)

        _type_check(data)
        data = _to_tensor(data)
        if len(data) > 0:
            buffer_ = _create_buffer(data)
            nested_size = _nested_size(data)
            nested_stride = _nested_stride(data)
            impl = _C._BufferNestedTensor(
                buffer_, nested_size, nested_stride)
        else:
            impl = _C._BufferNestedTensor(torch.empty(0), (), ())
        result = nested.NestedTensor(impl)

        if dtype is not None or device is not None:
            result = result.to(dtype=dtype, device=device)
        if requires_grad:
            result = result.requires_grad_(requires_grad)
        if pin_memory:
            result = result.pin_memory()
        return result
