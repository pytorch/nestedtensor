import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
import random

import utils
from utils import TestCase

class TestNestedTensorAutograd(TestCase):

    def test_grad(self):
        nt = nestedtensor.nested_tensor([torch.rand(1, 2), torch.rand(2, 3)], requires_grad=True)
        s = nt.sum()
        print(s)
        s.backward()
        print(nt.grad)
        print(nt)
        s = nt.sum((1, 2))
        print(s)
        s.backward()
        print(nt.grad)


        pass

if __name__ == "__main__":
    unittest.main()
