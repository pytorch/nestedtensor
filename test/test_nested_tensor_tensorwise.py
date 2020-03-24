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

class TestTensorWise(TestCase):

    def test_tensorwise(self):

        @nestedtensor.tensorwise()
        def simple_fn(t1, t2):
            return t1 + 1 + t2 * 2

        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nestedtensor.nested_tensor([a, b])
        nestedtensor.nested_tensor([b, a])

    def test_tensorwise_nested_dim_2(self):

        @nestedtensor.tensorwise()
        def simple_fn(t1, t2):
            return t1 + 1 + t2 * 2

        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nt1 = nestedtensor.nested_tensor([[a, b], [b]])
        nt2 = nestedtensor.nested_tensor([[b, a], [a]])
        simple_fn(nt1, nt2)

    def test_tensorwise_scalar(self):

        @nestedtensor.tensorwise(unbind_args=[2])
        def simple_fn_scalar(t1, t2, scalar):
            return t1 + scalar + t2 * 2

        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nt1 = nestedtensor.nested_tensor([a, b])
        nt2 = nestedtensor.nested_tensor([b, a])
        c = simple_fn_scalar(a, b, 2.0)
        nt3 = simple_fn_scalar(nt1, nt2, (2.0, 3.0))
        self.assertEqual(c, nt3[0])

    def test_tensorwise_tensor_kwarg(self):

        @nestedtensor.tensorwise(unbind_args=['out'])
        def simple_fn(t1, t2, t3=None):
            result = t1 * 2 + t2
            if t3 is not None:
                result = result + t3
            return result

        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nt1 = nestedtensor.nested_tensor([a, b])
        nt2 = nestedtensor.nested_tensor([b, a])
        c1 = simple_fn(a, b, t3=torch.tensor((0.5, 0.7)))
        c2 = simple_fn(b, a, t3=torch.tensor((0.5, 0.7)))
        nt3 = simple_fn(nt1, nt2, t3=torch.tensor((0.5, 0.7)))
        self.assertEqual(c2, nt3[1])
        self.assertEqual(c1, nt3[0])

if __name__ == "__main__":
    unittest.main()
