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

def _iter_constructors():
    yield nestedtensor.as_nested_tensor
    yield nestedtensor.nested_tensor

class TestFunctional(TestCase):
    def test_nll_loss(self):
        utils.gen_float_tensor(1, (40, 5))
        utils.gen_float_tensor(1, (40,))

    def test_addmm(self):
        torch.rand(5), torch.rand(4, 5)
        nestedtensor.nested_tensor(
            [torch.rand(1, 4), torch.rand(1, 4), torch.rand(4, 4)]
        )

    def test_conv2d(self):
        def _conv2d(input, *args, **kwargs):
            if input.dim() == 3:
                input = input.unsqueeze(0)
                result = torch.conv2d(input, *args, **kwargs)
                return result.squeeze(0)
            return torch.conv2d(input, *args, **kwargs)

        weight = torch.rand(64, 3, 7, 7)
        tf = nestedtensor.tensorwise()(_conv2d)
        images = [
            torch.rand(3, (i * 16) % 40 + 40, (i * 16) % 50 + 40) for i in range(128)
        ]
        nt = nestedtensor.nested_tensor(images)
        result = tf(nt, weight)
        result2 = torch.nn.functional.conv2d(nt, weight)
        for r, r2 in zip(result, result2):
            self.assertEqual(r, r2)

    def test_copy_(self):
        for constructor in _iter_constructors():
            nt1 = constructor([])
            nt2 = constructor([])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

            nt1 = constructor([torch.randn(1, 2, 3)])
            nt2 = constructor([torch.randn(1, 2, 3)])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

            nt1 = constructor([torch.randn(1, 2, 3), torch.randn(2, 1, 3)])
            nt2 = constructor([torch.randn(1, 2, 3), torch.randn(2, 1, 3)])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

            nt1 = constructor([[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            nt2 = constructor([[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

    def test_squeeze(self):
        for constructor in _iter_constructors():
            t = torch.randn(2, 3)
            result = constructor([t])
            nt = constructor([t.reshape(2, 3)])
            self.assertEqual(nt.squeeze(), result)
            nt = constructor([t.reshape(1, 2, 3)])
            self.assertEqual(nt.squeeze(), result)
            nt = constructor([t.reshape(1, 2, 1, 3, 1)])
            self.assertEqual(nt.squeeze(), result)
            nt = constructor([[t.reshape(1, 2, 1, 3)]])
            self.assertEqual(nt.squeeze(), result)
            nt = constructor([[[t.reshape(1, 2, 3)]]])
            self.assertEqual(nt.squeeze(), result)

            nt = constructor([t.reshape(1, 2, 3)])
            self.assertEqual(nt.squeeze(1), result)
            self.assertRaises(RuntimeError, lambda: nt.squeeze(0))
            self.assertRaises(RuntimeError, lambda: nt.squeeze(2))
            self.assertRaises(RuntimeError, lambda: nt.squeeze(3))
            self.assertRaises(IndexError, lambda: nt.squeeze(4))

            nt = constructor([[t.reshape(1, 2, 1, 3)]])
            self.assertEqual(nt.squeeze(1), constructor([t.reshape(1, 2, 1, 3)]))
            self.assertEqual(nt.squeeze(2), constructor([[t.reshape(2, 1, 3)]]))
            self.assertEqual(nt.squeeze(4), constructor([[t.reshape(1, 2, 3)]]))


if __name__ == "__main__":
    unittest.main()
