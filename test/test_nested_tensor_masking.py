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

class TestTensorMask(TestCase):
    # TODO: test comparison operator fails if nested_size doesn't agree.

    def test_simple(self):
        values = [torch.rand(i) for i in range(10)]
        random.shuffle(values)
        nt = nestedtensor.nested_tensor(values)
        tensor, mask = nt.to_tensor_mask()

    def test_nested(self):
        nt = utils.gen_nested_tensor(2, 2, 2)
        tensor, mask = nt.to_tensor_mask()

    def test_tensor_mask(self):
        nt = utils.gen_nested_tensor(2, 2, 2, size_low=1, size_high=2)
        tensor, mask = nt.to_tensor_mask()
        nt1 = nestedtensor.nested_tensor_from_tensor_mask(
            tensor, mask, nested_dim=nt.nested_dim())
        self.assertEqual(nt, nt1)
        nt2 = nestedtensor.nested_tensor_from_tensor_mask(
            tensor, mask)
        self.assertEqual(nt, nt2)

    def test_tensor_mask_lowdim(self):
        values = [torch.rand(1, 2) for i in range(10)]
        values = [values[1:i] for i in range(2, 10)]
        nt = nestedtensor.nested_tensor(values)
        tensor, mask = nt.to_tensor_mask()
        self.assertTrue([len(value) for value in values] == list(mask.sum(-1)))

if __name__ == "__main__":
    unittest.main()
