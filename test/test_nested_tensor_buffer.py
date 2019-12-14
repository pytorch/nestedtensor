import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from unittest import TestCase
import random

from nestedtensor._C import _BufferNestedTensor

import utils

# TODO: Test unbind, test grad and backward

class TestNestedTensorBuffer(TestCase):

    def test_default_constructor(self):
        # nested_dim is 1 and dim is 1 too.
        default_nested_tensor = _BufferNestedTensor(torch.tensor([]), [], [])
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


if __name__ == "__main__":
    unittest.main()
