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

def nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1):
    if tensor is None:
        raise RuntimeError("Tensor can't be undefined (None).")

    if mask is None:
        raise RuntimeError("Mask can't be undefined (None).")

    # Scalar was passed
    if tensor.dim() == 0:
        raise RuntimeError("Can't construct nested tensor from a scalar.")

    if nested_dim == 0:
        raise RuntimeError("Nested dimention can't be 0.")

    if mask.dim() == 0 and mask == False:
        raise RuntimeError("Scalar mask cant be False.")

    if nested_dim is not None and nested_dim > tensor.dim():
        raise RuntimeError("Nested dimention ({0}) can't be bigger than data tensor dimention ({1}).".format(nested_dim, tensor.dim()))

    if tensor.numel() == 0 and mask.numel() != 0:
        raise RuntimeError("Data tensor can't be emtpy if a mask has values.")

    if tensor.numel() != 0 and mask.numel() == 0:
        raise RuntimeError("Mask tensor can't be emtpy if a data tensor has values.")

    return nt_from_tensor_mask(tensor, mask, nested_dim, tensor.dtype)


def nt_from_tensor_mask(tensor, mask, nested_dim, dt):
    def _merge(tensors, nested_dim):
        if len(tensors) == 0:
            return torch.tensor([], dtype=dt)
        return torch.stack(tensors)

    if nested_dim == 0:
        if (mask.numel() == 0) or (mask.numel() == 1 and mask == True):
            return tensor

        if mask.dim() == 1:
            tensors = [tensor[i] if mask[i] else None for i in range(len(mask))]
            tensors = list(filter(lambda x: x is not None, tensors))
            return _merge(tensors, nested_dim)

        if mask.dim() > 1:
            tensors = [nt_from_tensor_mask(t, m, nested_dim, dt) for (t, m) in zip(tensor, mask)]
            if not all(t.numel() == 0 for t in tensors):
                tensors = list(filter(lambda x: x.numel() > 0, tensors))
            return _merge(tensors, nested_dim)
        else:
            return None
    else:
        inner_tensors = []
        if (mask.numel() == 0) or (mask.numel() == 1 and mask == True):
            for i in range(len(tensor)):
                inner_tensors.append(nt_from_tensor_mask(tensor[i], mask, nested_dim - 1, dt))
        elif (mask.numel() == 1 and mask == False):
            inner_tensors.append(None)
        else:
            inner_tensors = [nt_from_tensor_mask(t, m, nested_dim - 1, dt) for (t, m) in zip(tensor, mask)]

        # Filtering out None values which were ignored by mask
        inner_tensors = list(filter(lambda x: x is not None, inner_tensors))
        return creation.nested_tensor(inner_tensors)


def get_max_size_nt(obj, res=[1]):
    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            res = get_max_size_nt(o, res)

    if isinstance(obj, nestedtensor.nested.nested.NestedTensor):
        tres = get_max_size_nt(obj.unbind())
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


# TODO: mereg padding method with the one from get_result_tensor once NT.fill_ works
# as expected for NT([])
def get_result_mask(nt, shape):
    def pad_nt(nt, shape):
        res = []

        if isinstance(nt, torch.Tensor):
            if nt.numel() == 0:
                raise RuntimeError("Empty tensors are not yet supported.")

            # Dont pad in case of a scalar
            if nt.dim() == 0:
                return nt.item()

            return pad_tensor_to_shape(nt.new_full(nt.size(), True), shape).tolist()
        else:
            if len(nt) == 0:
                return [0]
            else:
                for entry in nt:
                    res.append(pad_nt(entry, shape))

        return res

    return torch.tensor(pad_nt(nt, shape), dtype=torch.bool, device=nt.device)


def get_result_tensor(nt, shape):
    def pad_nt(nt, shape):
        res = []

        if isinstance(nt, torch.Tensor):
            if nt.numel() == 0:
                raise RuntimeError("Empty tensors are not yet supported.")
            
            # Dont pad in case of a scalar
            if nt.dim() == 0:
                return nt.item()

            return pad_tensor_to_shape(nt, shape).tolist()
        else:
            if len(nt) == 0:
                return [0]
            else:
                for entry in nt:
                    res.append(pad_nt(entry, shape))

        return res

    return torch.tensor(pad_nt(nt, shape), dtype=nt.dtype, device=nt.device, requires_grad=nt.requires_grad)

# Return a tuple of a tensor and a mask that represent the given tensor list
# Returned tensor is always the same no matter what mask_dim was passed.
# If mask_dim was not passed, a mask with the smallest dimensionality would be returned.
# if passed mask_dim is lower than the minimal dimensionality of the mask that can represent 
# the data tensor, an error is thrown.
def to_tensor_mask(nt, mask_dim):
    if mask_dim is not None and mask_dim > nt.dim():
        raise RuntimeError("Mask dimention is bigger than nested dimention of a nested tensor.")

    # Check if scalar was passed
    if not isinstance(nt, list) and nt.size() == (1,):
        res_scalar = torch.tensor([nt[0].item()], dtype=nt.dtype, device=nt.device, requires_grad=nt.requires_grad)
        return res_scalar, torch.tensor(True)

    max_size = get_max_size_nt(nt)
    res_tensor = get_result_tensor(nt, max_size)
    res_mask = get_result_mask(nt, max_size)

    tensor_mask_tuple = merge_tensor_mask(TensorMask(res_tensor, res_mask), mask_dim)

    return tensor_mask_tuple.tensor, tensor_mask_tuple.mask


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
        collapsed_mask = normalize_mask(collapsed_mask)
        return merge_tensor_mask(TensorMask(tensor=tensor, mask=collapsed_mask), mask_dim)
    
    if mask_dim is not None and mask_dim != mask.dim():
        raise RuntimeError("Mask dimention is too small to represent data tensor.")
    return TensorMask(tensor=tensor, mask=mask)


# Convert mask values to boolean type
def normalize_mask(mask):
    assert (mask >= 0).sum() == mask.numel()
    mask = (mask > 0)
    return mask


def pad_tensor_to_shape(t, goal_shape):
    padd = ()
    tup = tuple(t.size())
    assert(t.dim() == len(goal_shape))
    for i in range(len(tup)):
        padd = (0, goal_shape[i] - tup[i]) + padd
    new_tensor = F.pad(t, padd)
    new_tensor = new_tensor.reshape(goal_shape)
    return new_tensor