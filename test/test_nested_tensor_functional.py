import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from unittest import TestCase
import random
import utils

class TestFunctional(TestCase):

    def test_nll_loss(self):
        a = utils.gen_float_tensor(1, (40, 5))
        b = utils.gen_float_tensor(1, (40,))

    def test_addmm(self):
        a, b = torch.rand(5), torch.rand(4, 5)
        nt = nestedtensor.nested_tensor(
            [torch.rand(1, 4), torch.rand(1, 4), torch.rand(4, 4)])

    def test_conv2d(self):
        def _conv2d(input, *args, **kwargs):
            if input.dim() == 3:
                input = input.unsqueeze(0)
                result = torch.conv2d(input, *args, **kwargs)
                return result.squeeze(0)
            return torch.conv2d(input, *args, **kwargs)

        weight = torch.rand(64, 3, 7, 7)
        tf = nestedtensor.tensorwise()(_conv2d)
        images = [torch.rand(3, (i * 16) % 40 + 40, (i * 16) % 50 + 40) for i in range(128)]
        nt = nestedtensor.nested_tensor(images)
        result = tf(nt, weight)
        result2 = torch.nn.functional.conv2d(nt, weight)
        for r, r2 in zip(result, result2):
            self.assertTrue((r == r2).all())

if __name__ == "__main__":
    unittest.main()
