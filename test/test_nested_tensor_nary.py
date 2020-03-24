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


class DynamicClassBase(TestCase):
    longMessage = True


def _gen_test_unary(func__, nested_dim, device):
    def _test_unary(self):
        data = utils.gen_nested_list(1, nested_dim, 3)
        data = utils.nested_map(lambda x: x.to(device), data)

        if func__ in ['log', 'log10', 'log2', 'rsqrt', 'sqrt']:
            data = utils.nested_map(lambda x: x.abs(), data)
        if func__ in ['acos', 'asin', 'erfinv', 'log1p']:
            data = utils.nested_map(lambda x: x.clamp(min=0, max=1), data)
        if func__ in ['mvlgamma']:
            data = utils.nested_map(lambda x: x.clamp(min=1), data)

        a1 = nestedtensor.nested_tensor(data)
        a3 = nestedtensor.nested_tensor(data)
        func_ = getattr(torch, func__)
        method_ = getattr(nestedtensor.NestedTensor, func__)
        method_inplace_ = getattr(nestedtensor.NestedTensor, func__ + "_")
        if func__ in ['clamp']:

            def func(x, out=None):
                return func_(x, min=-1, max=1, out=out)

            def method(x): return method_(x, min=-1, max=1)

            def method_inplace(x): return method_inplace_(x, min=-1, max=1)
        elif func__ in ['clamp_min']:

            def func(x, out=None):
                return func_(x, min=-1, out=out)

            def method(x): return method_(x, min=-1)

            def method_inplace(x): return method_inplace_(x, min=-1)
        elif func__ in ['clamp_max']:

            def func(x, out=None):
                return func_(x, 1, out=out)

            def method(x): return method_(x, 1)

            def method_inplace(x): return method_inplace_(x, 1)
        elif func__ in ['mvlgamma']:

            def func(x):
                return func_(x, p=2)

            def method(x): return method_(x, p=2)

            def method_inplace(x): return method_inplace_(x, p=2)
        elif func__ in ['renorm']:

            def func(x, out=None):
                return func_(x, 2, 0, 1.0, out=out)

            def method(x):
                return method_(x, 2, 0, 1.0)

            def method_inplace(x): return method_inplace_(x, 2, 0, 1.0)
        elif func__ in ['fmod']:

            def func(x, out=None):
                return func_(x, 0.3, out=out)

            def method(x): return method_(x, 0.3)

            def method_inplace(x): return method_inplace_(x, 0.3)
        else:
            func = func_
            method = method_
            method_inplace = method_inplace_

        a2 = nestedtensor.nested_tensor(utils.nested_map(func, data))

        self.assertTrue(a1.nested_dim() == a2.nested_dim())
        self.assertTrue(a2.nested_dim() == a3.nested_dim())

        def _close(t1, t2):
            self.assertAlmostEqual(t1, t2, ignore_contiguity=True)

        if func__ not in ['mvlgamma']:
            func(a1, out=a3)
            # TODO: Abstract this
            _close(func(a1), a3)
        _close(func(a1), a2)
        _close(method(a1), a2)
        _close(method_inplace(a1), a2)
        _close(a1, a2)
    return _test_unary


def _gen_test_binary(func):
    def _test_binary(self):
        a = utils.gen_float_tensor(1, (2, 3))
        b = utils.gen_float_tensor(2, (2, 3))
        c = utils.gen_float_tensor(3, (2, 3))
        # The constructor is supposed to copy!
        a1 = nestedtensor.nested_tensor([a, b])
        a2 = nestedtensor.nested_tensor([b, c])
        a1_l = nestedtensor.as_nested_tensor([a.clone(), b.clone()])
        a2_l = nestedtensor.as_nested_tensor([b.clone(), c.clone()])
        a3 = nestedtensor.nested_tensor([getattr(torch, func)(a, b),
                                  getattr(torch, func)(b, c)])
        a3_l = nestedtensor.as_nested_tensor(a3)
        self.assertEqual(a3_l, getattr(torch, func)(a1_l, a2_l))
        self.assertEqual(a3_l, getattr(torch, func)(a1, a2))
        self.assertEqual(a3, getattr(a1, func)(a2))
        self.assertEqual(a3, getattr(a1, func + "_")(a2))
        self.assertEqual(a3, a1)
    return _test_binary


TestUnary = type('TestUnary', (DynamicClassBase,), {})
for func__ in nestedtensor.nested.codegen.extension.get_unary_functions():
    if func__ == 'fill':
        continue
    for nested_dim in range(1, 5):
        avail_devices = ['cpu']
        if torch.cuda.is_available():
            avail_devices += ['cuda']
        for device in avail_devices:
            setattr(TestUnary, "test_{0}_nested_dim_{1}_{2}".format(
                func__, nested_dim, device), _gen_test_unary(func__, nested_dim, device))
TestBinary = type('TestBinary', (DynamicClassBase,), {})
for func in nestedtensor.nested.codegen.extension.get_binary_functions():
    setattr(TestBinary, "test_{0}".format(func),
            _gen_test_binary(func))

if __name__ == "__main__":
    unittest.main()
