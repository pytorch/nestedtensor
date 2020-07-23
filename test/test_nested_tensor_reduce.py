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

ntnt = nestedtensor.nested_tensor

class TestReduce(TestCase):

    def test_cumsum(self):
        t0 = torch.arange(9).reshape(3, 3)
        t1 = torch.arange(6).reshape(2, 3)
        t2 = torch.arange(9).reshape(3, 3)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)

        self.assertRaises(RuntimeError, lambda: torch.cumsum(nt, 0))
        self.assertRaises(RuntimeError, lambda: torch.cumsum(nt, 1))
        self.assertEqual(ntnt([[torch.cumsum(t0, 0), torch.cumsum(t1, 0)],
                               [torch.cumsum(t2, 0)]]), torch.cumsum(nt, 2))
        self.assertEqual(ntnt([[torch.cumsum(t0, 1), torch.cumsum(t1, 1)],
                               [torch.cumsum(t2, 1)]]), torch.cumsum(nt, 3))
        self.assertRaises(IndexError, lambda: torch.cumsum(nt, 4))


if __name__ == "__main__":
    unittest.main()
