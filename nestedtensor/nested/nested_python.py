import torch
from .layout import Layout

def _nn_functional_embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                                 scale_grad_by_freq=False, mode='mean', sparse=False,
                                 per_sample_weights=None, include_last_offset=False):
    # [...] Omitted input sanitization
    # [...] Verify that nested_size is shape compliant, i.e. all 1d Tensors (sequences)
    # Design decision: conversion happens automatically. This is similar to how we automatically
    # make Tensor contiguous or convert from fp16 to fp32 or sparse to dense if needed.
    # We could decide to throw a warning here.
    input = input.to(Layout.Packed)
    offsets = torch.tensor([0] + [x[0] for x in input.nested_size()[:-1]]).cumsum(0)
    # We could consider caching this metadata in NestedTensorPythonImpl
    offsets = offsets.to(data.device)
    assert input.layout is Layout.Packed
    return torch.nn.functional.embedding_bag(
        input.data,
        weight,
        offsets,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset)

def nested_tensor(tensors, dtype=None, device=None, requires_grad=False, layout=Layout.List): # pin_memory could be added as a layout
    """
    Given a list of Tensors, each of the same dimension but variable shape, construct a NestedTensorPythonImpl that represents
    this list of Tensors.

    If a given entry of tensors does not match the dtype or device of the others, the result dtype or device needs to
    be specified explicitly
    """
    assert layout is Layout.List # No other layout support for now
    assert isinstance(tensors, list)
    assert len(tensors) > 0
    dtype = tensors[0].dtype if dtype is None else dtype
    device = tensors[0].device if device is None else device
    # Change dtype and device if necessary
    tensors = [t.to(device, dtype) for t in tensors]
    nested_size = tuple(x.size() for x in tensors)
    return NestedTensorPythonImpl(tensors, nested_size, Layout.List, dtype, device, requires_grad).to(layout)

def _from_packed_sequence_to_list(packed_sequence):
    padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True)
    tensors = []
    for i, length in enumerate(lengths):
        tensors.append(padded[i, :length])
    return tensors

def as_nested_tensor(data, dtype=None, device=None, requires_grad=False, layout=Layout.List): # pin_memory could be added as a layout
    """
    Similar to torch.as_tensor, this converts the given data into a NestedTensorPythonImpl.
    """
    if isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return nested_tensor(_from_packed_sequence_to_list(data))
    raise NotImplementedError("as_nested_tensor cannot convert data of type {} into a NestedTensorPythonImpl.".format(type(data)))


def _from_list_to_layout(list_nt, target_layout):
    assert list_nt.layout is Layout.List
    if target_layout is Layout.List:
        return list_nt
    if target_layout is Layout.Masked:
        max_size = [len(list_nt.data)]
        for d in range(list_nt.data[0].dim()):
            max_size.append(max(x.size(d) for x in list_nt.data))
        # This approach doesn't support autograd and can also be used during construction or without autograd
        # An approach that does work with autograd uses pad and cat, but is a bit more involved
        # See https://github.com/pytorch/NestedTensorPythonImpl/blob/master/NestedTensorPythonImpl/nested/masking.py#L142 for a complete implementation
        data = torch.zeros(*max_size, dtype=list_nt.dtype, device=list_nt.device)
        mask = torch.zeros(*max_size, dtype=torch.bool, device=list_nt.device)
        for d_t, d_m, t in zip(data, mask, list_nt.data):
            for d in range(t.dim()):
                d_t = d_t.narrow(d, 0, t.size(d))
                d_m = d_m.narrow(d, 0, t.size(d))
            d_t.copy_(t.detach())
            d_m.fill_(1)
        return NestedTensorPythonImpl(data, list_nt.nested_size(), Layout.Masked, list_nt.dtype, list_nt.device, list_nt.requires_grad, metadata=mask)
    if target_layout is Layout.Packed:
        offsets_ = list_nt.nested_size()
        data = torch.cat([x.reshape(-1) for x in list_nt.data]) # shape information is stored in nested_size
        return NestedTensorPythonImpl(data, list_nt.nested_size(), Layout.Packed, list_nt.dtype, list_nt.device, list_nt.requires_grad)
    if target_layout is Layout.PackedSequence:
        return NestedTensorPythonImpl(torch.nn.utils.rnn.pack_sequence(list_nt.data, enforce_sorted=False), # enforce_sorted set to False doesn't support ONNX for now,
                            list_nt.nested_size(),
                            Layout.PackedSequence,
                            list_nt.dtype,
                            list_nt.device,
                            list_nt.requires_grad)
    raise NotImplemented("Converstion from list to target layout {} not supported.".format(target_layout.name))
            
class NestedTensorPythonImpl(object):
    def __init__(self, data, nested_size, layout, dtype, device, requires_grad, metadata=None):
        # Can be list of tensors, single packed or masked Tensor or PackedSequence
        self.data = data
        # Metadata is overloaded with type and meaning
        # Masked: Stores bool mask where True means included, False means excluded
        # Packed: Stores 1d Tensor of offsets. offsets are the length of each entry in the flat data. Packed currently only supports 2d NestedTensorPythonImpls
        # PackedSequence: Stores the lengths of the PackedSequence
        self.metadata = metadata
        self._nested_size = nested_size
        self._layout = layout
        self._dtype = dtype
        self._device = device
        # Gradient is supported by differentiable layout conversion functions a tracked by data field
        self._requires_grad = requires_grad 

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.nn.functional.embedding_bag:
            # Design decision pending: We could make conversion to Layout.Padding automatic
            return _nn_functional_embedding_bag(*args, **kwargs)
        raise NotImplementedError("Given func {} does not support NestedTensorPythonImpl.".format(func))

    def nested_size(self):
        return self._nested_size

    @property
    def dtype(self):
        return self._dtype

    @property
    def layout(self):
        return self._layout

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad

    # There are 5 layouts, therefore there are 20 possible
    # conversions excluding identities
    def to(self, target_layout):
        assert isinstance(target_layout, Layout)
        if self.layout is target_layout:
            return self
        if self.layout is Layout.List:
            return _from_list_to_layout(self, target_layout)
        raise NotImplementedError(
            "Cannot convert {} to desired layout {}".format(
                self.layout.name, target_layout.name))

    
    def to_tensor_list(self):
        # Returns a list of Tensors
        return self.to(Layout.List).data

    def to_padded(self, padding_value=-1):
        # Returns a Tensor padded with padding_value
        converted = self.to(Layout.Masked)
        return converted.data.masked_fill_(~converted.metadata, padding_value)

    def to_masked(self):
        # Returns a Tensor plus a Bool mask of same shape
        converted = self.to(Layout.Masked)
        return converted.data, converted.mask

    def to_packed_sequence(self):
        return self.to(Layout.PackedSequence).data
              
