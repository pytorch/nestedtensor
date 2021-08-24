import torch.fx as fx
from typing import Type, Dict, Any, Tuple, Iterable
import torch
import copy
from torch.fx import symbolic_trace
import time

def my_add_relu(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    return y.add_(x).relu_()

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
        # NOTE: This will be faster once https://github.com/pytorch/pytorch/pull/62482 lands
        if not self.slow_fusion and inp.is_contiguous(memory_format=torch.contiguous_format):
            inp = inp.to(memory_format=torch.channels_last)
        if self.slow_fusion and inp.is_contiguous(memory_format=torch.channels_last):
            inp = inp.to(memory_format=torch.contiguous_format)
        if not self.slow_fusion and not self.weight_is_channels_last:
            self.weight.data = self.weight.to(memory_format=torch.channels_last)
            inp = inp.to(memory_format=torch.channels_last)
            self.weight_is_channels_last = True
        # NOTE: Very hacky way of dealing with cudnn_convolution_relu's inability
        # to support contiguous weight but channels last input. We also
        # can't just set all weights to channels last in this model, because
        # the first layer is very slow under channels last.
        try:
            return torch.cudnn_convolution_relu(inp,
                                                self.weight,
                                                self.bias,
                                                self.stride,
                                                self.padding,
                                                self.dilation,
                                                self.groups)
        except RuntimeError:
            if self.weight.is_contiguous(memory_format=torch.channels_last):
                self.weight.data = self.weight.to(memory_format=torch.contiguous_format)
            else:
                self.weight.data = self.weight.to(memory_format=torch.channels_last)

        return torch.cudnn_convolution_relu(inp,
                                            self.weight,
                                            self.bias,
                                            self.stride,
                                            self.padding,
                                            self.dilation,
                                            self.groups)


class Conv2dAddReLU(torch.nn.Module):
    def __init__(self,
                 weight,
                 bias,
                 stride,
                 padding,
                 dilation,
                 groups):
        super(Conv2dAddReLU, self).__init__()
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

    def forward(self, inp, add_input):
        # TODO: Reactivate this once cudnn_convolution_add_relu is fixed.
        # weight = self.weight.to(memory_format=torch.contiguous_format)
        # if not self.slow_fusion and inp.is_contiguous(memory_format=torch.contiguous_format):
        #     inp = inp.to(memory_format=torch.channels_last)
        #     add_input = add_input.to(memory_format=torch.channels_last)
        # if self.slow_fusion and inp.is_contiguous(memory_format=torch.channels_last):
        #     inp = inp.to(memory_format=torch.contiguous_format)
        #     add_input = add_input.to(memory_format=torch.contiguous_format)
        # if not self.slow_fusion and not self.weight_is_channels_last:
        #     self.weight.data = self.weight.to(memory_format=torch.channels_last)
        #     inp = inp.to(memory_format=torch.channels_last)
        #     add_input = add_input.to(memory_format=torch.channels_last)
        #     self.weight_is_channels_last = True
        # return torch.cudnn_convolution_add_relu(inp,
        #                                         self.weight,
        #                                         add_input,
        #                                         1.0,
        #                                         self.bias,
        #                                         self.stride,
        #                                         self.padding,
        #                                         self.dilation,
        #                                         self.groups)
        out = torch.conv2d(inp,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.groups)
        my_add_relu(add_input, out)
        # out.add_(add_input)
        # out.relu_()
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


    last_nodes = []
    count = 0
    for node in new_graph.nodes:
        if count == 31:
            break
        if (node.op == "call_function" or node.op == "call_module"):
            last_nodes.append(node)
            if len(last_nodes) == 4:
                last_nodes = last_nodes[1:]
        if len(last_nodes) < 3:
            continue
        is_match = True
        is_match = is_match and (last_nodes[0].op == "call_module")
        is_match = is_match and (last_nodes[1].op == "call_function")
        is_match = is_match and (last_nodes[2].op == "call_module")
        is_match = is_match and isinstance(modules[last_nodes[0].target], torch.nn.Conv2d)
        is_match = is_match and (str(last_nodes[1]).split("_")[0] == "add")
        is_match = is_match and isinstance(modules[last_nodes[2].target], torch.nn.ReLU)
        if (is_match):
            conv = modules[last_nodes[1].args[0].target]
            fused_conv = Conv2dAddReLU(conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
            replace_node_module(last_nodes[2], modules, fused_conv)
            last_nodes[2].args = (last_nodes[0].args[0], last_nodes[1].args[1])
            new_graph.erase_node(last_nodes[1])
            new_graph.erase_node(last_nodes[0])
            count += 1
    return fx.GraphModule(fx_model, new_graph)


def fuse_conv_add_relu(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    new_graph.lint()
    return fx.GraphModule(fx_model, new_graph)
