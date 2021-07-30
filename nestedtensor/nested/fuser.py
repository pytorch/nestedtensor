import torch.fx as fx
from typing import Type, Dict, Any, Tuple, Iterable
import torch
import copy
from torch.fx import symbolic_trace
import time

def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

# Works for length 2 patterns with 2 modules
def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)

def computeUpdatedConvWeightAndBias(
        bn_rv,
        bn_eps,
        bn_w,
        bn_b,
        bn_rm,
        conv_w,
        conv_b=None):
    orig_dtype = bn_rv.dtype
    bn_var_rsqrt = (bn_w / torch.sqrt(bn_rv.to(torch.double) + bn_eps))
    new_w = (conv_w * (bn_var_rsqrt).reshape(-1, 1, 1, 1)).to(orig_dtype)
    if conv_b is None:
        return new_w
    new_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return new_w, new_b

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)
    fused_conv.bias = None

    fused_conv.weight = \
        torch.nn.Parameter(computeUpdatedConvWeightAndBias(bn.running_var, bn.eps, bn.weight, bn.bias, bn.running_mean, fused_conv.weight))

    return fused_conv

def fuse_conv_bn(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(torch.nn.Conv2d, torch.nn.BatchNorm2d)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)

class Conv2dReLU(torch.nn.Module):
    def __init__(self,
                 weight,
                 bias,
                 stride,
                 padding,
                 dilation,
                 groups):
        super(Conv2dReLU, self).__init__()
        self.weight = weight
        self.weight_is_channels_last = False
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.slow_fusion = False
        if self.weight.size(2) == 7 and self.weight.size(3) == 7:
            self.slow_fusion = True

    def forward(self, inp):
        if not self.slow_fusion and inp.is_contiguous(memory_format=torch.contiguous_format):
            inp = inp.to(memory_format=torch.channels_last)
        if self.slow_fusion and inp.is_contiguous(memory_format=torch.channels_last):
            inp = inp.to(memory_format=torch.contiguous_format)
        if not self.slow_fusion and not self.weight_is_channels_last:
            self.weight.data = self.weight.to(memory_format=torch.channels_last)
            inp = inp.to(memory_format=torch.channels_last)
            self.weight_is_channels_last = True
        out = torch.cudnn_convolution_relu(inp,
                                            self.weight,
                                            self.bias,
                                            self.stride,
                                            self.padding,
                                            self.dilation,
                                            self.groups)
        return out

def fuse_conv_relu(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(torch.nn.Conv2d, torch.nn.ReLU)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                relu = modules[node.target]
                fused_conv = Conv2dReLU(conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)
