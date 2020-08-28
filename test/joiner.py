import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from utils import TestCase
import random
import utils
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torch import nn
import math

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: nestedtensor.NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            # import torchviz
            # dot = torchviz.make_dot(x.sum())
            # dot.format = 'svg'
            # dot.render('asdf.svg')
            out.append(x)
            print(self[1](x).dtype)
            pos.append(self[1](x))

        return out, pos
