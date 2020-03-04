import torch
import torch.nn.functional as F
import numbers
import collections

from . import creation
from . import utils
import nestedtensor

TensorMask = collections.namedtuple('TensorMask', 'tensor mask')

def nested_tensor_from_padded_tensor(tensor, nested_dim=None, padding=-1):
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
        raise RuntimeError("Nested dimension ({0}) can't be bigger than data tensor dimension ({1}).".format(nested_dim, tensor.dim()))

    if tensor.numel() == 0 and mask.numel() != 0:
        raise RuntimeError("Data tensor can't be emtpy if a mask has values.")

    if tensor.numel() != 0 and mask.numel() == 0:
        raise RuntimeError("Mask tensor can't be emtpy if a data tensor has values.")

    return nt_from_tensor_mask(tensor, mask, nested_dim)


def nt_from_tensor_mask(tensor, mask, nested_dim):
    def _merge(tensors, nested_dim):
        if len(tensors) == 0:
            return torch.tensor([], dtype=tensor.dtype, device=tensor.device, requires_grad=tensor.requires_grad)
        return torch.stack(tensors)

    if nested_dim == 0:
        if (mask.numel() == 0) or (mask.numel() == 1 and mask.item() == True):
            return tensor

        if mask.dim() == 1:
            tensors = [tensor[i] if mask[i] else None for i in range(len(mask))]
            tensors = list(filter(lambda x: x is not None, tensors))
            return _merge(tensors, nested_dim)

        if mask.dim() > 1:
            tensors = [nt_from_tensor_mask(t, m, nested_dim) for (t, m) in zip(tensor, mask)]
            if not all(t.numel() == 0 for t in tensors):
                tensors = list(filter(lambda x: x.numel() > 0, tensors))
            return _merge(tensors, nested_dim)
        else:
            return None
    else:
        inner_tensors = []
        if (mask.numel() == 0) or (mask.numel() == 1 and mask == True):
            for i in range(len(tensor)):
                inner_tensors.append(nt_from_tensor_mask(tensor[i], mask, nested_dim - 1))
        elif (mask.numel() == 1 and mask == False):
            inner_tensors.append(None)
        else:
            inner_tensors = [nt_from_tensor_mask(t, m, nested_dim - 1) for (t, m) in zip(tensor, mask)]

        # Filtering out None values which were ignored by mask
        inner_tensors = list(filter(lambda x: x is not None, inner_tensors))
        return creation.nested_tensor(inner_tensors)

# Get max size per each dimension from all the passed tensors.
def get_max_size(obj, res=[1]):
    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            res = get_max_size(o, res)

    if isinstance(obj, nestedtensor.nested.nested.NestedTensor):
        tres = get_max_size(obj.unbind())
        while len(tres) > len(res):
                res.append(0)

        res = [max(i, j) for (i, j) in zip(res, tres)]

    if isinstance(obj, torch.Tensor):
        # scalar
        if obj.dim() == 0 and obj.numel() == 1:
            res = [1]
        else:
            while len(obj.size()) > len(res):
                res.append(0)

            res = [max(i, j) for (i, j) in zip(res, obj.size())]

    return res

def get_tensor_mask(nt, shape):
    def pad_nt(nt, shape):
        res_tensor = []
        res_mask = []

        if isinstance(nt, torch.Tensor):
            if nt.numel() == 0:
                raise RuntimeError("Empty tensors are not yet supported.")

            # Dont pad in case of a scalar
            if nt.dim() == 0:
                return nt.item(), nt.item()

            tensor = pad_tensor_to_shape(nt, shape).tolist()
            mask = pad_tensor_to_shape(nt.new_full(nt.size(), True), shape).tolist()
            return tensor, mask
        else:
            if len(nt) == 0:
                return [0], [0]
            else:
                for entry in nt:
                    tensor, mask = pad_nt(entry, shape)
                    res_tensor.append(tensor)
                    res_mask.append(mask)

        return res_tensor, res_mask

    t, m = pad_nt(nt, shape)
    tensor = torch.tensor(t, dtype=nt.dtype, device=nt.device, requires_grad=nt.requires_grad)
    mask = torch.tensor(m, dtype=torch.bool, device=nt.device)
    return tensor, mask


# Return a tuple of a tensor and a mask that represent the given tensor list
# Returned tensor is always the same no matter what mask_dim was passed.
# If mask_dim was not passed, a mask with the smallest dimensionality would be returned.
# if passed mask_dim is lower than the minimal dimensionality of the mask that can represent 
# the data tensor, an error is thrown.
def to_tensor_mask(nt, mask_dim):
    if mask_dim is not None and mask_dim > nt.dim():
        raise RuntimeError("Mask dimension is bigger than nested dimension of a nested tensor.")

    # Check if scalar was passed
    if not isinstance(nt, list) and nt.size() == (1,):
        res_scalar = torch.tensor([nt[0].item()], dtype=nt.dtype, device=nt.device, requires_grad=nt.requires_grad)
        mask = torch.tensor(True) if mask_dim == 0 or mask_dim == None else torch.tensor([True])
        return res_scalar, mask

    max_size = get_max_size(nt)
    res_tensor, res_mask = get_tensor_mask(nt, max_size)
    tensor_mask_tuple = merge_tensor_mask(TensorMask(res_tensor, res_mask), mask_dim)

    return tensor_mask_tuple.tensor, tensor_mask_tuple.mask


# Merge mask to a given dimension if possible.
def merge_tensor_mask(tensor_mask, mask_dim):
    tensor = tensor_mask.tensor
    mask = tensor_mask.mask
    if mask_dim is not None and mask.dim() == mask_dim:
        return tensor_mask

    if mask.dim() == 0:
        return tensor_mask

    last_size = mask.size(-1)
    collapsed_mask = mask.sum(-1)
    is_last_size = (collapsed_mask == last_size)
    is_zero = (collapsed_mask == 0)
    if (is_last_size.sum() + is_zero.sum()) == collapsed_mask.numel():
        collapsed_mask = collapsed_mask.to(torch.bool)
        return merge_tensor_mask(TensorMask(tensor=tensor, mask=collapsed_mask), mask_dim)

    if mask_dim is not None and mask_dim != mask.dim():
        raise RuntimeError("Mask dimension is too small to represent data tensor.")
    return TensorMask(tensor=tensor, mask=mask)


def pad_tensor_to_shape(t, goal_shape):
    padd = ()
    tup = tuple(t.size())
    assert(t.dim() == len(goal_shape))
    for i in range(len(tup)):
        padd = (0, goal_shape[i] - tup[i]) + padd
    new_tensor = F.pad(t, padd)
    new_tensor = new_tensor.reshape(goal_shape)
    return new_tensor
