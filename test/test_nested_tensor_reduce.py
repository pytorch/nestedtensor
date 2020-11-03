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


def _flatten_list(ts):
    if not isinstance(ts, list):
        return [ts]
    return sum(map(_flatten_list, ts), [])


class TestReduce(TestCase):

    def _test_reduce_dim(self, fn, associative=True):
        t0 = torch.arange(9).float().reshape(3, 3)
        t1 = torch.arange(6).float().reshape(2, 3)
        t2 = torch.arange(9).float().reshape(3, 3)
        ts = [[t0, t1], [t2, t1]]
        nt = ntnt(ts)
        if associative:
            t01 = fn(torch.stack([fn(t0, 0), fn(t1, 0)]), 0)
            t21 = fn(torch.stack([fn(t2, 0), fn(t1, 0)]), 0)
            t02 = fn(torch.stack([fn(t0, 0), fn(t2, 0)]), 0)
            t11 = fn(torch.stack([fn(t1, 0), fn(t1, 0)]), 0)
            self.assertEqual(ntnt([t01, t21]), fn(nt, (1, 2)))
            self.assertEqual(ntnt([t02, t11]), fn(nt, (0, 2)))

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
        self._test_reduce_dim(torch.cumsum, False)

    def _test_allreduce(self, fn, with_grad=False):
        def test(ts):
            if with_grad:
                nt = ntnt(ts)
            else:
                nt = nestedtensor.nested_tensor(ts)
            print("--")
            print(nt)
            t = fn(nt)
            print(t)
            print("--")
            # , fn(t2)])
            a = torch.cat([x.reshape(-1) for x in _flatten_list(ts)])
            print(a)
            print(fn(a))
            self.assertEqual(t, fn(a))
            fn(a).backward()
            if with_grad:
                t.backward()
                # TODO: Re-enable under autograd
                self.assertEqual(nt.grad[0][0], t0.grad)
                self.assertEqual(nt.grad[0][1], t1.grad)
                self.assertEqual(nt.grad[1][0], t2.grad)

        t0 = torch.randn(3, 3, requires_grad=True)
        t1 = torch.randn(2, 3, requires_grad=True)
        t2 = torch.randn(3, 3, requires_grad=True)
        # t0 = torch.arange(2 * 1).reshape(2, 1).float()
        # t1 = torch.arange(2 * 1).reshape(2, 1).float() + t0.numel()
        # t1 = t1 * 2
        # t2 = torch.arange(2 * 1).reshape(2, 1).float() + t1.numel() + t0.numel()
        # t2 = t2 * 4
        # t0.requires_grad_()
        # t1.requires_grad_()
        # t2.requires_grad_()
        test([t0])
        test([t0, t1])
        test([t0, t1, t2])
        test([[t0], [t1]])
        test([[t0]])
        test([[t0, t1], [t2]])
        test([[t0, t1, t2]])

    def test_sum(self):
        self._test_allreduce(lambda x: x.sum(), True)
        self._test_reduce_dim(torch.sum)

    def test_mean(self):
        self._test_allreduce(lambda x: x.mean())
        self._test_reduce_dim(torch.mean)

    def test_prod(self):
        self._test_allreduce(lambda x: x.prod())

    def test_var(self):
        self._test_allreduce(lambda x: x.var(unbiased=False))
        self._test_allreduce(lambda x: x.var(unbiased=True))


if __name__ == "__main__":
    unittest.main()
