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
from torch import nn
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        not_mask = []
        for tensor in tensor_list:
            not_mask.append((torch.ones_like(tensor, dtype=torch.bool).prod(0)).bool())
        not_mask = nestedtensor.nested_tensor(not_mask, dtype=torch.bool, device=tensor.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats)
        dim_t = nestedtensor.nested_tensor(len(tensor_list) * [dim_t], dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos = nestedtensor.cat((pos_y, pos_x), dim=3)
        pos_sin = pos[:, :, :, 0::2].sin()
        pos_cos = pos[:, :, :, 1::2].cos()
        res = nestedtensor.stack((pos_sin, pos_cos), dim=4)
        return res.flatten(3)
