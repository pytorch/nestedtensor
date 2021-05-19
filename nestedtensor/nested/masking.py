import torch
import torch.nn.functional as F
import numbers
import collections

from . import creation
import nestedtensor

TensorMask = collections.namedtuple('TensorMask', 'tensor mask')


def nested_tensor_from_padded_tensor(tensor, nested_dim=1, padding=-1):
    mask = (tensor != padding)
    return nested_tensor_from_tensor_mask(tensor, mask, nested_dim)


# Constructs nested tensor from passed tensor and mask.
def nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1):
    if tensor is None:
        raise RuntimeError("Tensor can't be undefined (None).")

    if mask is None:
        raise RuntimeError("Mask can't be undefined (None).")

    # Scalar was passed
    if tensor.dim() == 0:
        raise RuntimeError("Can't construct nested tensor from a scalar.")

    if nested_dim == 0:
        raise RuntimeError("Nested dimension can't be 0.")

    if nested_dim is not None and nested_dim > tensor.dim():
        raise RuntimeError("Nested dimension ({0}) can't be bigger than data tensor dimension ({1}).".format(
            nested_dim, tensor.dim()))

    if tensor.numel() == 0 and mask.numel() != 0:
        raise RuntimeError("Data tensor can't be emtpy if a mask has values.")

    if tensor.numel() != 0 and mask.numel() == 0:
        raise RuntimeError(
            "Mask tensor can't be emtpy if a data tensor has values.")

    return nt_from_tensor_mask(tensor, mask, nested_dim)


def nt_from_tensor_mask(tensor, mask, nested_dim):
    result = torch.ops.nestedtensor.nt_from_tensor_mask(
        tensor, mask, nested_dim)
    assert result is not None
    return nestedtensor.NestedTensor(result).contiguous()
