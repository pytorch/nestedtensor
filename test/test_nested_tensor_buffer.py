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

# TODO: Test unbind, test grad and backward

class TestNestedTensorBuffer(TestCase):

    def test_default_constructor(self):
        # nested_dim is 1 and dim is 1 too.
        default_nested_tensor = nestedtensor._C._buffer_nested_tensor(torch.tensor([]), [], [])
        default_tensor = torch.tensor([])
        self.assertEqual(default_nested_tensor.nested_dim(), 1)
        self.assertEqual(default_nested_tensor.nested_size(), [])
        self.assertEqual(default_nested_tensor.nested_stride(), [])
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(default_nested_tensor.requires_grad,
                         default_tensor.requires_grad)
        self.assertEqual(default_nested_tensor.is_pinned(),
                         default_tensor.is_pinned())

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
