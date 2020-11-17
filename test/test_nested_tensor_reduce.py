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


def _flatten_nt(nt):
    if not isinstance(nt, nestedtensor.NestedTensor):
        return [nt]
    return sum(map(_flatten_nt, nt.unbind()), [])


class TestReduce(TestCase):

    def _test_reduce_dim(self, fn, associative=True):
        t0 = torch.arange(9).float().reshape(3, 3)
        t1 = torch.arange(6).float().reshape(2, 3)
        t2 = torch.arange(9).float().reshape(3, 3)
        t3 = torch.arange(3).float().reshape(3, 1)
        t4 = torch.arange(3).float().reshape(1, 3)
        ts = [[t0, t1], [t2, t1]]
        nt = ntnt(ts)
        if associative:
            t01 = fn(torch.stack([fn(t0, 0), fn(t1, 0)]), 0)
            t21 = fn(torch.stack([fn(t2, 0), fn(t1, 0)]), 0)
            t02 = fn(torch.stack([fn(t0, 0), fn(t2, 0)]), 0)
            t11 = fn(torch.stack([fn(t1, 0), fn(t1, 0)]), 0)
            self.assertEqual(ntnt([t01, t21]), fn(nt, (1, 2)))
            self.assertEqual(ntnt([t02, t11]), fn(nt, (0, 2)))

        ts = [t3, t4]
        nt = ntnt(ts)
        print(nt)
        print(fn(nt, 0))

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
            t = fn(nt)
            flat_ts = _flatten_list(ts)
            a = torch.cat([x.reshape(-1) for x in flat_ts])
            a_res = fn(a)
            # print("_0_")
            # print(t)
            # print(a_res)
            self.assertEqual(t, a_res)
            if with_grad:
                a_res.backward()
                t.backward()
                nt_grads = _flatten_nt(nt.grad)
                for a, b in zip(nt_grads, flat_ts):
                    # print(a)
                    # print(b.grad)
                    # print("--")
                    self.assertEqual(a, b.grad)

        def gen_ts():
            t0 = torch.randn(4, 3, requires_grad=True)
            t1 = torch.randn(2, 3, requires_grad=True)
            t2 = torch.randn(3, 4, requires_grad=True)
            t3 = torch.randn(3, 4, requires_grad=True)
            t4 = torch.randn(3, 4, requires_grad=True)
            return t0, t1, t2, t3, t4

        t0, t1, t2, t3, t4 = gen_ts()
        test([t0])
        t0, t1, t2, t3, t4 = gen_ts()
        test([t0, t1])
        t0, t1, t2, t3, t4 = gen_ts()
        test([t0, t1, t2])
        t0, t1, t2, t3, t4 = gen_ts()
        test([t0, t1, t2, t3])
        t0, t1, t2, t3, t4 = gen_ts()
        test([[t0], [t1, t2]])
        t0, t1, t2, t3, t4 = gen_ts()
        test([[t0, t1], [t2]])
        t0, t1, t2, t3, t4 = gen_ts()
        test([[t0, t1], [t2, t3]])
        t0, t1, t2, t3, t4 = gen_ts()
        test([[t0, t1], [t2, t3], [t4]])

    def test_sum_all(self):
        self._test_allreduce(lambda x: x.sum(), True)

    def test_sum_dim(self):
        self._test_reduce_dim(torch.sum, True)

    def test_mean_all(self):
        self._test_allreduce(lambda x: x.mean())

    def test_mean_dim(self):
        self._test_reduce_dim(torch.mean, True)

    def test_prod(self):
        self._test_allreduce(lambda x: x.prod())

    def test_var(self):
        self._test_allreduce(lambda x: x.var(unbiased=False), True)
        self._test_allreduce(lambda x: x.var(unbiased=True))

    def test_sum_to(self):
        a = ntnt([torch.randn(1, 2), torch.randn(2, 1)])
        b = ntnt([torch.randn(1), torch.randn(1)])
        print(a)
        print(nestedtensor.nested.nested.sum_to(a._impl, a.nested_size()))
        print(nestedtensor.nested.nested.sum_to(a._impl, b.nested_size()))
        print(nestedtensor.nested.nested.sum_to(a._impl, [1, 2]))
        print(a)
        print(nestedtensor.nested.nested.sum_to(a._impl, [2]))
        pass


if __name__ == "__main__":
    unittest.main()
