"""
This file contains functions to overwrite or extend functions, methods or functionals such as
torch.nn.functional.conv2d or torch.Tensor.addmm or torch.relu
"""

import torch
import torch.nn.functional as F
import numbers
from functools import wraps
import collections
import os

from . import utils
from . import nested

from nestedtensor import _C

from numbers import Number

orig_squeeze = torch.squeeze


def squeeze(*args, **kwargs):
    if utils.find_nested_tensor_dispatch_key(*args) is None:
        return orig_squeeze(*args, **kwargs)
    return utils.tensorwise(dim_args=[1, 'dim'], wrap_dim_args=False)(orig_squeeze)(*args, **kwargs)


orig_unsqueeze = torch.squeeze


def unsqueeze(*args, **kwargs):
    if utils.find_nested_tensor_dispatch_key(*args) is None:
        return orig_unsqueeze(*args, **kwargs)
    return utils.tensorwise(dim_args=[1, 'dim'], wrap_dim_args=False)(orig_unsqueeze)(*args, **kwargs)


orig_linear = torch.nn.functional.linear


def linear(input, weight, bias=None):
    # TODO: what if bias is a NestedTensor?
    if utils.find_nested_tensor_dispatch_key(input, weight) is None:
        return orig_linear(input, weight, bias)

    if (utils.is_nested_tensor(input) and torch.is_tensor(weight) and
            input.is_contiguous() and weight.is_contiguous() and
            None not in input.size() and None not in weight.size() and
            (bias is None or
             (bias.is_contiguous() and None not in bias.size() and torch.is_tensor(bias)))):
        input_buffer = input._impl.get_buffer().view(input.size())
        output = torch.matmul(input_buffer, weight.t())
        if bias is not None:
            output = output + bias
        return output

    return utils.tensorwise()(orig_linear)(input, weight, bias)


orig_conv2d = torch.nn.functional.conv2d


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # TODO: Dispatch on bias NestedTensor too?
    if utils.find_nested_tensor_dispatch_key(input, weight) is None:
        return orig_conv2d(input, weight, bias, stride, padding, dilation, groups)

    def _conv2d(input, *args, **kwargs):
        if input.dim() == 3:
            input = input.unsqueeze(0)
            result = orig_conv2d(input, *args, **kwargs)
            return result.squeeze(0)
        return orig_conv2d(input, *args, **kwargs)

    return utils.tensorwise()(_conv2d)(input, weight, bias, stride, padding, dilation, groups)


orig_max_pool2d = torch.nn.functional.max_pool2d


def max_pool2d(input, *args, **kwargs):
    # TODO: Support more NestedTensor arguments?
    if utils.find_nested_tensor_dispatch_key(input) is None:
        return orig_max_pool2d(input, *args, **kwargs)

    def _max_pool2d(input, *args, **kwargs):
        # Support for unbatched image
        if input.dim() == 3:
            input = input.unsqueeze(0)
            result = orig_max_pool2d(input, *args, **kwargs)
            return result.squeeze(0)
        return orig_max_pool2d(input, *args, **kwargs)

    return utils.tensorwise()(_max_pool2d)(input, *args, **kwargs)


orig_embedding_bag = torch.nn.functional.embedding_bag


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                  scale_grad_by_freq=False, mode='mean', sparse=False,
                  per_sample_weights=None):
    # TODO: Not all cases work with EmbeddingBag yet
    if utils.find_nested_tensor_dispatch_key(input, weight) is None:
        return orig_embedding_bag(input, weight, offsets, max_norm, norm_type,
                                  scale_grad_by_freq, mode, sparse,
                                  per_sample_weights)

    def _offsets(nested_size):
        if isinstance(nested_size[0], torch.Size):
            return list(x[0] for x in nested_size)
        else:
            return sum([_offsets(x) for x in nested_size], [])

    def _new_nested_size(nested_size, embed_dim):
        if isinstance(nested_size[0], torch.Size):
            new_size = torch.Size((embed_dim,))
            return [new_size] * len(nested_size)
        else:
            return [_new_nested_size(x, embed_dim) for x in nested_size]

    # Adding support for NestedTensors of 1d sequences and
    # weight being a Tensor.

    def _embedding_bag(input, *args, **kwargs):
        if 'offsets' not in kwargs and input.dim() == 1:
            input = input.unsqueeze(0)
            result = orig_embedding_bag(input, *args, **kwargs)
            return result.squeeze(0)
        return orig_embedding_bag(input, *args, **kwargs)

    # Special case for performance
    if utils.is_nested_tensor(input):
        if (input.is_contiguous() and input.dim() - input.nested_dim() == 1 and
                torch.is_tensor(weight) and offsets is None):
            offsets = (torch.tensor([0] + _offsets(input.nested_size()))
                       .cumsum(0)[:-1].to(input.device, non_blocking=True))
            buffer_ = orig_embedding_bag(input._impl.get_buffer(), weight, offsets, max_norm, norm_type,
                                         scale_grad_by_freq, mode, sparse, per_sample_weights)
            buffer_ = buffer_.view(-1)
            new_nested_size = _new_nested_size(
                input.nested_size(), weight.size(1))
            return nested.NestedTensor(
                _C._BufferNestedTensor(
                    buffer_, new_nested_size))

    return utils.tensorwise()(_embedding_bag)(input, weight, offsets, max_norm, norm_type,
                                              scale_grad_by_freq, mode, sparse,
                                              per_sample_weights)


orig_batch_norm = torch.nn.functional.batch_norm


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    if utils.find_nested_tensor_dispatch_key(input, running_mean, running_var, weight, bias, training, momentum, eps) is None:
        return orig_batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

    if utils.is_nested_tensor(input) and input.is_contiguous() and None not in input.size():
        # TODO: The unsqueeze and squeeze needs to depend on input dimension
        input_buffer = input._impl.get_buffer().reshape(input.size())
        result = orig_batch_norm(
            input_buffer, running_mean, running_var, weight, bias, training, momentum, eps)
        return nested.NestedTensor(_C._BufferNestedTensor(result.flatten(), input.nested_size()))

    def t_batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps):
        squeeze_after = False
        # TODO: Need support for BatchNorm1d and BatchNorm2d as well
        if input.dim() == 3:
            # Check if is single image (leading batch dimension removed)
            if input.size(1) != running_mean.size(0):
                input = input.unsqueeze(0)
                squeeze_after = True
        result = orig_batch_norm(
            input, running_mean, running_var, weight, bias, training, momentum, eps)
        if squeeze_after:
            result = result.squeeze(0)
        return result

    tw_batch_norm = utils.tensorwise()(t_batch_norm)

    return tw_batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)


orig_nll_loss = torch.nn.functional.nll_loss


# TODO: Return scalar?
def nll_loss(input, target, *args, **kwargs):
    if utils.is_nested_tensor(input):
        loss = None
        for i in range(len(input)):
            loss_i = orig_nll_loss(input[i].unsqueeze(
                0), target[i].unsqueeze(0), *args, **kwargs)
            if loss is None:
                loss = loss_i
            else:
                loss += loss_i
        return loss
    else:
        return orig_nll_loss(input, target, *args, **kwargs)


orig_lstm_forward = torch.nn.modules.rnn.LSTM.forward


def lstm_forward(self, input, hx=None):
    # TODO: In the future nesting can be supported if hx is nested.
    # TODO: Write this to wrap packed sequence
    if utils.find_nested_tensor_dispatch_key(input) is None:
        return orig_lstm_forward(self, input, hx)

    if not input.nested_dim() == 1:
        raise ValueError(
            "Only accepting NestedTensors of nested_dim 1 for now.")
    if not self.batch_first:
        raise ValueError("NestedTensore requires batch_first to be True")
    result = []
    for i in range(len(input)):
        o, (h, c) = orig_lstm_forward(
            self, input[i].unsqueeze(0), (hx[0].narrow(1, i, 1), hx[1].narrow(1, i, 1)))
        o = o.squeeze(0)
        result.append((o, (h, c)))
    output = torch.as_nested_tensor([o for (o, h) in result])
    hidden0 = torch.cat([h[0] for (o, h) in result], dim=1)
    hidden1 = torch.cat([h[1] for (o, h) in result], dim=1)
    return output, (hidden0, hidden1)


orig_interpolate = torch.nn.functional.interpolate


def interpolate(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None):
    if utils.find_nested_tensor_dispatch_key(input) is None:
        return orig_interpolate(input, size, scale_factor, mode, align_corners)

    def _interpolate(input, size, scale_factor, mode, align_corners):
        # TODO: Document this
        squeeze_after = False
        if input.dim() == 3:
            input = input.unsqueeze(0)
            squeeze_after = True
        result = orig_interpolate(input, size,
                                  scale_factor, mode, align_corners)
        if squeeze_after:
            result = result.squeeze(0)
        return result

    tf = utils.tensorwise(unbind_args=[1, 'size'])(_interpolate)
    return tf(input, size, scale_factor, mode, align_corners)


def _set_size(nested_size, dim, size):
    if isinstance(nested_size, torch.Size):
        result = list(nested_size)
        result[dim] = size
        return torch.Size(tuple(result))
    return list(map(lambda x: _set_size(x, dim - 1, size), nested_size))


def mm(*args, **kwargs):
    if (utils.is_nested_tensor(args[0]) and
            args[0].size(-1) is not None and
            args[0].is_contiguous() and
            torch.is_tensor(args[1])):
        self = args[0]
        weight = args[1]
        result = self.flatten().view(-1, self.size(-1)).mm(weight)
        result_nested_size = _set_size(
            self.nested_size(), self.dim() - 1, result.size(-1))
        buffer_ = result.flatten()
        return nested.NestedTensor(
            _C._BufferNestedTensor(buffer_, result_nested_size))

    tf = utils.tensorwise()(torch.Tensor.mm)
    return tf(*args, **kwargs)


def _addmm(*args, **kwargs):
    import pdb
    pdb.set_trace()
    tf = utils.tensorwise()(torch.addmm)
    return tf(*args, **kwargs)


def addmm(*args, **kwargs):
    TensorType = (torch.Tensor, nested.NestedTensor)
    if utils.match_type_signature_prefix((Number, TensorType, Number, TensorType, TensorType), args):
        return _addmm(*args, **kwargs)
    elif utils.match_type_signature_prefix((TensorType, Number, TensorType, TensorType), args):
        return _addmm(1, *args, **kwargs)
    elif utils.match_type_signature_prefix((Number, TensorType, TensorType, TensorType), args):
        return _addmm(args[0], args[1], 1, *args[2:], **kwargs)
    elif utils.match_type_signature_prefix((TensorType, TensorType, TensorType), args):
        return _addmm(1, args[0], 1, *args[1:], **kwargs)
    else:
        raise ValueError("Unrecognized signature for addmm")
