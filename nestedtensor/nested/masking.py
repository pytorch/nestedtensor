import torch
import torch.nn.functional as F
import numbers

import collections

from . import creation

TensorMask = collections.namedtuple('TensorMask', 'tensor mask')

# TODO: Can use masked_select?

def nested_tensor_from_padded_tensor(tensor, nested_dim=None, padding=-1):
    mask = (tensor != padding)
    return nested_tensor_from_tensor_mask(tensor, mask, nested_dim)

"""
Given a tensor t of size N_1 x N_2 x ... N_n and mask m of size M_1 x M_2 x ... M_d where d <= n
return a NestedTensor of dimensionality d where subtensor t[i_1][i_2]...[i_d] is included as a Tensor constituent if m[i_1][i_2]...[i_d].
The resulting NestedTensor will be of dimension n.
Unless specific, it is attempted to yield a NestedTensor of minimal nested dimension.
"""
def nested_tensor_from_tensor_mask(tensor, mask, nested_dim=None):
    # TODO: Should be possible with views only.
    if not mask.dim() > 0:
        raise RuntimeError("Mask has to have dimention > 0")

    # TODO: Need to define empty-list and 0 numel semantics
    # TODO: Test edge-cases and kwargs
    def _merge(tensors, nested_dim=None):
        assert len(tensors) > 0

        def _mergable_tensors(tensors, nested_dim):
            if nested_dim is None:
                size0 = tensors[0].size()
                compliant = all(torch.is_tensor(t) for t in tensors)
                compliant = all(size0 == t.size()
                                for t in tensors) and compliant
                return compliant
            else:
                return nested_dim < 1

        if _mergable_tensors(tensors, nested_dim):
            return torch.stack(tensors)
        else:
            return creation.nested_tensor(tensors)

    if nested_dim is not None and mask.dim() < nested_dim:
        raise ValueError("Mask dimension ({0}) is too small to construct nested tensor with nested dimension of {1}".format(mask.dim(), nested_dim))

    if mask.dim() == 1:
        tensors = [tensor[i] if mask[i]
                   else None for i in range(len(mask))]
        tensors = list(filter(lambda x: x is not None, tensors))
        if len(tensors) == 0:
            return tensor.narrow(0, 0, 0)
        return _merge(tensors, nested_dim)
    else:
        sub_nested_dim = None if nested_dim is None else nested_dim - 1
        tensors = [nested_tensor_from_tensor_mask(
            t, m, sub_nested_dim) for (t, m) in zip(tensor, mask)]
        if not all(t.numel() == 0 for t in tensors):
            tensors = list(filter(lambda x: x.numel() > 0, tensors))
        if len(tensors) == 0:
            return tensor.narrow(0, 0, 0)
        return _merge(tensors, nested_dim)


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
    return TensorMask(tensor=tensor, mask=mask)


def tensor_list(lst):
    def _tensor_list(l):
        if isinstance(l, list):
            impls = []
            for entry in l:
                impls.append(_tensor_list(entry))
            return stack(impls)
        else:
            assert isinstance(l, torch.Tensor)
            return TensorMask(tensor=l, mask=torch.ones_like(l).to(torch.bool))

    assert isinstance(lst, list), "Is " + str(type(lst))
    assert len(lst) > 0
    return _tensor_list(lst)


def make_tensor_mask(tensor_lst, mask_dim):
    tensor_mask = tensor_list(tensor_lst)
    tensor_mask = merge_tensor_mask(tensor_mask, mask_dim)
    return tensor_mask.tensor, tensor_mask.mask


def pad_tensor_to_shape(t, goal_shape):
    padd = ()
    tup = tuple(t.size())
    assert(t.dim() == len(goal_shape))
    for i in range(len(tup)):
        padd = (0, goal_shape[i] - tup[i]) + padd
    new_tensor = F.pad(t, padd)
    new_tensor = new_tensor.reshape(goal_shape)
    return new_tensor


def _cat(nts, skip_empty):
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

    assert len(nts) > 0, "Can't concatenate less than 1 Tensors"
    # It makes no sense to concatenate a number to something
    for nt in nts:
        assert(nt.tensor.dim() > 0)
    # For now we only support the concatenation of
    for i, nt in enumerate(nts):
        if i + 1 < len(nts):
            assert(nt.tensor.dim() == nts[i + 1].tensor.dim())

    max_tensor_shape = _max_shape([tuple(nt.tensor.size()) for nt in nts])
    max_mask_shape = _max_shape([tuple(nt.mask.size()) for nt in nts])

    tensors = []
    masks = []
    all_zero_numel = True
    # TODO: This fails for mixed tensors
    for i in range(len(nts)):
        # Skip empty tensors akin to torch.cat
        if nts[i].tensor.numel() > 0 and skip_empty:
            continue
        all_zero_numel = False
        assert nts[i].tensor.size() == nts[i].mask.size()
        tensor = pad_tensor_to_shape(nts[i].tensor, max_tensor_shape)
        mask = pad_tensor_to_shape(nts[i].mask, max_mask_shape)
        tensors.append(tensor)
        masks.append(mask)

    assert not all_zero_numel

    tensor = torch.cat(tensors)
    mask = torch.cat(masks)
    return TensorMask(tensor=tensor, mask=mask)


# All shapes, but first dim must match
# Empty or not, doesn't matter
def stack(nts_):
    if len(nts_) == 0:
        raise RuntimeError("expected a non-empty list of TensorMasks")
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
    return _cat(nts, False)


# All shapes must match. Deprecated behavior supports insane catting
# of non-shape matching empty tensors but we don't want that. Don't support this here and
# throw an Error.
def cat(nts):
    return _cat(nts, True)


def normalize_mask(mask):
    assert (mask >= 0).sum() == mask.numel()
    mask = (mask > 0)
    return mask


def check_mask(mask):
    assert (mask.numel() == ((mask == 0).sum() +
                             (mask == 1).sum()))
