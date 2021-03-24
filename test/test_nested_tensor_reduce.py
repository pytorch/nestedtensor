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

from nestedtensor.nested.nested import native_is_expandable_to


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

    def _test_reduce_dim(self, fn, associative=True, test_keep_dim=True, test_multi_dim=True):
        t0 = torch.arange(9).float().reshape(3, 3)
        t1 = torch.arange(6).float().reshape(2, 3)
        t2 = torch.arange(9).float().reshape(3, 3)
        ts = [[t0, t1], [t2, t1]]
        nt = ntnt(ts)
        if associative and test_multi_dim:
            t01 = fn(torch.stack([fn(t0, 0), fn(t1, 0)]), 0)
            t21 = fn(torch.stack([fn(t2, 0), fn(t1, 0)]), 0)
            t02 = fn(torch.stack([fn(t0, 0), fn(t2, 0)]), 0)
            t11 = fn(torch.stack([fn(t1, 0), fn(t1, 0)]), 0)
            self.assertEqual(ntnt([t01, t21]), fn(nt, (1, 2)))
            self.assertEqual(ntnt([t02, t11]), fn(nt, (0, 2)))

            if test_keep_dim:
                t01 = fn(torch.stack([fn(t0, 0), fn(t1, 0)]), 0, True)
                t21 = fn(torch.stack([fn(t2, 0), fn(t1, 0)]), 0, True)
                t02 = fn(torch.stack([fn(t0, 0), fn(t2, 0)]), 0, True)
                t11 = fn(torch.stack([fn(t1, 0), fn(t1, 0)]), 0, True)
                self.assertEqual(ntnt([[t01, t21]]), fn(nt, (1, 2), True))
                self.assertEqual(ntnt([[t02, t11]]), fn(nt, (0, 2), True))

        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self.assertRaises(RuntimeError, lambda: fn(nt, 0))
        self.assertRaises(RuntimeError, lambda: fn(nt, 1))
        self.assertEqual(ntnt([[fn(t0, 0), fn(t1, 0)],
                               [fn(t2, 0)]]), fn(nt, 2))
        self.assertEqual(ntnt([[fn(t0, 1), fn(t1, 1)],
                               [fn(t2, 1)]]), fn(nt, 3))
        if test_keep_dim:
            self.assertEqual(ntnt([[fn(t0, 0, True), fn(t1, 0, True)],
                                   [fn(t2, 0, True)]]), fn(nt, 2, True))
            self.assertEqual(ntnt([[fn(t0, 1, True), fn(t1, 1, True)],
                                   [fn(t2, 1, True)]]), fn(nt, 3, True))
        self.assertRaises(IndexError, lambda: fn(nt, 4))

    @unittest.skip("Requires autograd support")
    def test_cumsum(self):
        self._test_reduce_dim(torch.cumsum, False, False)

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

    @unittest.skip("Requires autograd support")
    def test_sum_all(self):
        self._test_allreduce(lambda x: x.sum(), True)

    @unittest.skip("Requires autograd support")
    def test_sum_dim(self):
        self._test_reduce_dim(torch.sum, True)

    @unittest.skip("Requires autograd support")
    def test_max_all(self):
        self._test_allreduce(lambda x: x.max())

    @unittest.skip("Requires autograd support")
    def test_max_dim(self):
        self._test_reduce_dim(lambda x, dim, keepdim=False: x.max(
            dim, keepdim)[0], True, test_multi_dim=False)

    @unittest.skip("Requires autograd support")
    def test_mean_all(self):
        self._test_allreduce(lambda x: x.mean())

    @unittest.skip("Requires autograd support")
    def test_mean_dim(self):
        self._test_reduce_dim(torch.mean, True)

    @unittest.skip("Requires autograd support")
    def test_prod(self):
        self._test_allreduce(lambda x: x.prod())

    @unittest.skip("Requires autograd support")
    def test_var(self):
        self._test_allreduce(lambda x: x.var(unbiased=False), True)
        self._test_allreduce(lambda x: x.var(unbiased=True))

    @unittest.skip("Requires autograd support")
    def test_var_dim(self):
        t0 = torch.arange(9).float().reshape(3, 3)
        t1 = torch.arange(6).float().reshape(2, 3)
        t2 = (torch.arange(9).float().reshape(3, 3) - 9).pow(2)
        t0 = torch.randn(3, 3)
        t1 = torch.randn(2, 3)
        t2 = torch.randn(3, 3)
        t3 = torch.randn(2, 3)

        ts = [t0, t1]
        nt = ntnt(ts)
        res = torch.var(nt, 1)
        self.assertEqual(
            ntnt([torch.var(t0, 0), torch.var(t1, 0)]), res)
        self.assertRaises(RuntimeError, lambda: res.sum().backward())

        res = torch.var(nt, 2)
        self.assertEqual(
            ntnt([torch.var(t0, 1), torch.var(t1, 1)]), res)
        self.assertRaises(RuntimeError, lambda: res.sum().backward())

        ts = [t0, t2]
        nt = ntnt(ts)
        res = torch.var(nt, 0)
        self.assertEqual(torch.stack(ts).var(0), res)
        self.assertRaises(RuntimeError, lambda: res.sum().backward())

        res = torch.var(nt, 1)
        self.assertEqual(
            ntnt([torch.var(t0, 0), torch.var(t2, 0)]), res)
        self.assertRaises(RuntimeError, lambda: res.sum().backward())

        res = torch.var(nt, 2)
        self.assertEqual(
            ntnt([torch.var(t0, 1), torch.var(t2, 1)]), res)
        self.assertRaises(RuntimeError, lambda: res.sum().backward())

        self.assertEqual(torch.stack(ts).var(
            (0, 1), unbiased=False), torch.var(nt, (0, 1), unbiased=False))

        nt = ntnt([t0, t1])
        self.assertRaisesRegex(
            RuntimeError, "Can only reduce across nested dimensions of Tensor compliant shapes.", lambda: torch.var(nt, 0))

        nt = ntnt([[t0, t1], [t2, t3]])
        self.assertRaisesRegex(
            RuntimeError, "Can only reduce across nested dimension 0.", lambda: torch.var(nt, 1))
        self.assertRaisesRegex(
            RuntimeError, "Can only reduce across nested dimensions if given nested tensor is of nested dimension 1.", lambda: torch.var(nt, 0))
        t0_var0 = torch.var(t0, 0)
        t1_var0 = torch.var(t1, 0)
        t2_var0 = torch.var(t2, 0)
        t3_var0 = torch.var(t3, 0)
        self.assertEqual(
            ntnt([[t0_var0, t1_var0], [t2_var0, t3_var0]]), torch.var(nt, 2))
        t0_var1 = torch.var(t0, 1)
        t1_var1 = torch.var(t1, 1)
        t2_var1 = torch.var(t2, 1)
        t3_var1 = torch.var(t3, 1)
        self.assertEqual(
            ntnt([[t0_var1, t1_var1], [t2_var1, t3_var1]]), torch.var(nt, 3))

    @unittest.skip("Requires autograd support")
    def test_sum_to_size(self):
        a = ntnt([torch.arange(2).reshape(1, 2),
                  torch.arange(2).reshape(2, 1) + 2])
        # b = ntnt([torch.randn(1), torch.randn(1)])
        # print(a)
        # print(nestedtensor.nested.nested.sum_to(a._impl, a.nested_size()))
        # print(nestedtensor.nested.nested.sum_to(a._impl, b.nested_size()))
        # print(nestedtensor.nested.nested.sum_to(a._impl, [1, 2]))
        print(a)
        # print(nestedtensor.nested.nested.sum_to(a, (2,)))
        # print(nestedtensor.nested.nested.sum_to(a, (2, 2)))
        a = ntnt([torch.arange(2).reshape(1, 2),
                  torch.arange(2).reshape(1, 2) + 2])
        b = ntnt([torch.arange(2).reshape(2),
                  torch.arange(2).reshape(2) + 2])
        print(nestedtensor.nested.nested.sum_to_size(a, a))
        print('a')
        print(a)
        print(nestedtensor.nested.nested.sum_to_size(a, b))
        # self.assertRaises(
        #     RuntimeError, lambda: nestedtensor.nested.nested.sum_to_size(a, b))
        self.assertRaises(RuntimeError, lambda: nestedtensor.nested.nested.sum_to_size(
            torch.randn(1, 2), a))
        print(nestedtensor.nested.nested.sum_to_size(a, torch.randn(1, 2)))
        print(nestedtensor.nested.nested.sum_to_size(a, torch.randn(1, 2)).shape)
        # b = ntnt([torch.randn(1), torch.randn(1)])
        pass

    @unittest.skip("Requires autograd support")
    def test_native_is_expandable_to(self):
        a = ntnt([torch.arange(2).reshape(1, 2),
                  torch.arange(2).reshape(1, 2) + 2])
        self.assertEqual(True, native_is_expandable_to(a, a))
        self.assertEqual(False, native_is_expandable_to(a, torch.randn(1, 2)))
        self.assertEqual(True, native_is_expandable_to(torch.randn(1, 2), a))
        self.assertEqual(True, native_is_expandable_to(torch.randn(2), a))
        self.assertEqual(False, native_is_expandable_to(torch.randn(2, 1), a))
        b = ntnt([torch.arange(2).reshape(2),
                  torch.arange(2).reshape(2) + 2])
        c = ntnt([[torch.arange(2).reshape(1, 2)],
                  [torch.arange(2).reshape(1, 2) + 2]])
        # Both NT
        self.assertEqual(True, native_is_expandable_to(b, a))
        self.assertEqual(False, native_is_expandable_to(a, b))
        self.assertEqual(True, native_is_expandable_to(a, c))
        self.assertEqual(False, native_is_expandable_to(c, a))
        # Shape NT, desired T
        pass

    @unittest.skip("Requires autograd support")
    def test_sizes_equal(self):
        a = ntnt([torch.arange(2).reshape(1, 2),
                  torch.arange(2).reshape(1, 2) + 2])
        b = ntnt([torch.arange(2).reshape(2),
                  torch.arange(2).reshape(2) + 2])
        self.assertEqual(True, nestedtensor.nested.nested.sizes_equal(a, a))
        self.assertEqual(False, nestedtensor.nested.nested.sizes_equal(a, b))
        self.assertEqual(False, nestedtensor.nested.nested.sizes_equal(b, a))
        self.assertEqual(
            False, nestedtensor.nested.nested.sizes_equal(torch.randn(1, 2), a))
        self.assertEqual(
            False, nestedtensor.nested.nested.sizes_equal(a, torch.randn(1, 2)))
        self.assertEqual(True, nestedtensor.nested.nested.sizes_equal(
            torch.randn(1, 2), torch.randn(1, 2)))
        self.assertEqual(False, nestedtensor.nested.nested.sizes_equal(
            torch.randn(2, 1), torch.randn(1, 2)))
        pass


if __name__ == "__main__":
    unittest.main()
