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

class TestFunctional(TestCase):

    def test_nll_loss(self):
        a = utils.gen_float_tensor(1, (40, 5))
        b = utils.gen_float_tensor(1, (40,))

    def test_addmm(self):
        a, b = torch.rand(5), torch.rand(4, 5)
        nt = nestedtensor.nested_tensor(
            [torch.rand(1, 4), torch.rand(1, 4), torch.rand(4, 4)])

    def test_conv2d(self):
        tensor1 = torch.rand(3, 128, 128)
        tensor2 = torch.rand(3, 128, 128) 
        list_of_tensors = [tensor1, tensor2]

        weight = torch.rand(3, 3, 7, 7)
        nt = nestedtensor.nested_tensor(list_of_tensors)
        nt_res = [t for t in torch.nn.functional.conv2d(nt, weight).unbind()]
        tensor_res = [torch.nn.functional.conv2d(t.unsqueeze(0), weight).squeeze(0) for t in list_of_tensors]
        self.assertEqual(nt_res, tensor_res)

if __name__ == "__main__":
    unittest.main()
