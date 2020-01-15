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

# TODO: Test unbind, test grad and backward

class TestNestedTensorBuffer(TestCase):

    def test_grad(self):
        nt = nestedtensor.nested_tensor([torch.rand(1, 2)])
        nt.requires_grad_(True)
        a = nt.unbind()[0]
        c = nt.sum()
        c.backward()
        # An unbound Tensor does not accumulate gradients because it's a
        # partial view of the buffer.
        self.assertIsNone(a.grad)
        nt_grad = nt.grad
        self.assertIs(nt._impl.get_buffer().grad, nt_grad._impl.get_buffer())
        # Unbinding the gradient is legitimate for further processing.
        self.assertIsNotNone(nt_grad.unbind()[0])

    # TODO
    def test_detach(self):
        pass


if __name__ == "__main__":
    unittest.main()
