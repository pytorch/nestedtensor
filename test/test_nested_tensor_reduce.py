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

def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)

class TestReduce(TestCase):

    def _test_reduce_dim(self, fn):
        t0 = torch.arange(9).float().reshape(3, 3)
        t1 = torch.arange(6).float().reshape(2, 3)
        t2 = torch.arange(9).float().reshape(3, 3)
        ts = [[t0, t1], [t2, t1]]
        nt = nestedtensor.nested_tensor(ts)

        print('nt')
        print(nt)
        print('-1-')
        print(fn(nt, 2))
        print(fn(nt, (1, 2)))
        # print('-2-')
        # print(fn(nt, (0, 2)))
        # print('-3-')
        # print(fn(nt, (0, 1, 2)))
        # print('-4-')
        # print(fn(nt))
        # print('-5-')
        # print(t0.sum() + t1.sum() + t2.sum() + t1.sum())

        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        self.assertRaises(RuntimeError, lambda: fn(nt, 0))
        self.assertRaises(RuntimeError, lambda: fn(nt, 1))
        self.assertEqual(nestedtensor.nested_tensor([[fn(t0, 0), fn(t1, 0)],
                               [fn(t2, 0)]]), fn(nt, 2))
        self.assertEqual(nestedtensor.nested_tensor([[fn(t0, 1), fn(t1, 1)],
                               [fn(t2, 1)]]), fn(nt, 3))
        self.assertRaises(IndexError, lambda: fn(nt, 4))

    def test_cumsum(self):
        self._test_reduce_dim(torch.cumsum)

    def _test_allreduce(self, fn, with_grad=False):
        t0 = torch.randn(3, 3, requires_grad=True)
        t1 = torch.randn(2, 3, requires_grad=True)
        t2 = torch.randn(3, 3, requires_grad=True)
        ts = [[t0, t1], [t2]]
        # nt = nestedtensor.nested_tensor(ts) #, requires_grad=True)
        if with_grad:
            nt = ntnt(ts)
        else:
            nt = nestedtensor.nested_tensor(ts)
        t = fn(nt)
        a = torch.stack([fn(t0), fn(t1), fn(t2)])
        self.assertEqual(t, fn(a))
        fn(a).backward()
        if with_grad:
            t.backward()
            # TODO: Re-enable under autograd
            self.assertEqual(nt.grad[0][0], t0.grad)
            self.assertEqual(nt.grad[0][1], t1.grad)
            self.assertEqual(nt.grad[1][0], t2.grad)

    def test_sum(self):
        self._test_allreduce(lambda x: x.sum(), True)
        self._test_reduce_dim(torch.sum)

    def test_mean(self):
        self._test_allreduce(lambda x: x.mean())
        self._test_reduce_dim(torch.mean)

    def test_prod(self):
        self._test_allreduce(lambda x: x.prod())


if __name__ == "__main__":
    unittest.main()
