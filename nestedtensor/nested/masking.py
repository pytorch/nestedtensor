import torch
import torch.nn.functional as F
import numbers
import collections

from . import creation
from . import utils

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
        tensor_lst = nt
    else:
        tensor_lst = nt.to_list()

    # check if a scalar was passed
    if not isinstance(tensor_lst, list) and tensor_lst.size() == (1,):
        res_scalar = torch.tensor([tensor_lst[0].item()], dtype=nt.dtype, device=nt.device, requires_grad=nt.requires_grad)
        return res_scalar, torch.tensor(True)
    
    assert isinstance(tensor_lst, list), "A scalar or a list was expected. Please, report this error."
    
    tensor_mask_tuple = make_tensor_mask_tuple(tensor_lst, nt)
    tensor_mask_tuple = merge_tensor_mask(tensor_mask_tuple, mask_dim)
    return tensor_mask_tuple.tensor, tensor_mask_tuple.mask


def make_tensor_mask_tuple(tensor_list, nt):
    if isinstance(tensor_list, list):
        impls = []
        for entry in tensor_list:
            impls.append(make_tensor_mask_tuple(entry, nt))
        return stack(impls, nt)
    else:
        assert isinstance(tensor_list, torch.Tensor)
        
        # throw an error in case of an empty tensor
        if tensor_list.numel() == 0:
            raise RuntimeError("Empty tensors are not yet supported.")

        return TensorMask(tensor=tensor_list, mask=torch.ones_like(tensor_list, dtype=torch.bool))


def merge_tensor_mask(tensor_mask, mask_dim):
    tensor = tensor_mask.tensor
    mask = tensor_mask.mask
    if mask_dim is not None and mask.dim() == mask_dim:
        return tensor_mask
    
    if mask.dim() == 0:
        if mask:
            return tensor_mask
        else:
            raise ValueError("If mask is 0-dim, it must be True")
    
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


def _cat(nts):
    if len(nts) == 0:
        raise RuntimeError("expected a non-empty list of TensorMasks")

    def _max_shape(tups):
        if len(tups) == 0:
            return ()

        result = len(tups[0]) * [0]
        for i in range(len(tups)):
            for j in range(len(result)):
                result[j] = max(result[j], tups[i][j])

        return tuple(result)


    # It makes no sense to concatenate a number to something
    for nt in nts:
        assert(nt.tensor.dim() > 0), "Can't concatenate non-tensor elements"

    # For now we only support the concatenation of tensors with the same dimentionality
    for i, nt in enumerate(nts):
        if i + 1 < len(nts):
            if nt.tensor.dim() != nts[i + 1].tensor.dim():
                raise RuntimeError("Can't concatenate tensors of different dimentionality.")

    max_tensor_shape = _max_shape([tuple(nt.tensor.size()) for nt in nts])
    max_mask_shape = _max_shape([tuple(nt.mask.size()) for nt in nts])

    tensors = []
    masks = []
    for i in range(len(nts)):
        assert nts[i].tensor.size() == nts[i].mask.size()
        tensor = pad_tensor_to_shape(nts[i].tensor, max_tensor_shape)
        mask = pad_tensor_to_shape(nts[i].mask, max_mask_shape)
        tensors.append(tensor)
        masks.append(mask)

    tensor = torch.cat(tensors)
    mask = torch.cat(masks)
    return TensorMask(tensor=tensor, mask=mask)


# All shapes, but first dim must match
# Empty or not, doesn't matter
def stack(nts_, nt):
    nts = []

    # Raise dimensionality by 1
    for entry in nts_:
        new_tensor = entry.tensor
        new_tensor_shape = (1, ) + tuple(new_tensor.size())
        new_tensor = new_tensor.reshape(new_tensor_shape)

        new_mask = entry.mask
        new_mask_shape = (1, ) + tuple(new_mask.size())
        new_mask = new_mask.reshape(new_mask_shape)

        nts.append(TensorMask(tensor=new_tensor, mask=new_mask))

    # empty NT
    if len(nts_) == 0:
        nts.append(TensorMask(mask=torch.tensor([], dtype=torch.bool, device=nt.device),
                              tensor=torch.tensor([], dtype=nt.dtype,
                                                      device=nt.device,
                                                      requires_grad=nt.requires_grad)))

    return _cat(nts)
