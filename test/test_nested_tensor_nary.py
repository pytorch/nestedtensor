import torch
import nestedtensor
import unittest
from utils import TestCase
from utils import get_unary_functions
from utils import get_binary_functions
from utils import get_python_binary_arithmetic_operations
import utils


def ntnt(x, device=None):
    return nestedtensor.nested_tensor(
        x, requires_grad=False, device=device)


def ntnt_nograd(x, device=None):
    return nestedtensor.nested_tensor(x, device=device)


class DynamicClassBase(TestCase):
    longMessage = True


def _gen_test_unary(func__, nested_dim, device):
    def _test_unary(self):
        data = utils.gen_nested_list(1, nested_dim, 3, size_high=1)
        data = utils.nested_map(lambda x: x.to(device), data)

        if func__ in ['log', 'log10', 'log2', 'rsqrt', 'sqrt']:
            data = utils.nested_map(lambda x: x.abs(), data)
        if func__ in ['acos', 'asin', 'erfinv', 'log1p']:
            data = utils.nested_map(lambda x: x.clamp(min=0, max=1), data)
        if func__ in ['mvlgamma']:
            data = utils.nested_map(lambda x: x.clamp(min=1), data)

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

        def _close(t1, t2):
            self.assertAlmostEqual(t1, t2, ignore_contiguity=True)

        a1 = ntnt(data, device=device)
        a2 = ntnt(
            utils.nested_map(func, data), device=device)
        _close(func(a1), a2)
        _close(method(a1), a2)

        a1 = ntnt_nograd(data, device=device)
        a2 = ntnt_nograd(
            utils.nested_map(func, data), device=device)
        a3 = ntnt_nograd(data, device=device)

        self.assertEqual(a1.nested_dim(), a2.nested_dim())
        self.assertEqual(a2.nested_dim(), a3.nested_dim())

        if func__ not in ['mvlgamma']:
            func(a1, out=a3)
            # TODO: Abstract this
            _close(func(a1), a3)
        _close(method_inplace(a1), a2)
        _close(a1, a2)
    return _test_unary


def _gen_test_binary(func, no_grad):
    def _test_binary(self):
        a = utils.gen_float_tensor(1, (2, 3))#  * 0 + 1
        b = utils.gen_float_tensor(2, (2, 3))#  * 0 + 2
        c = utils.gen_float_tensor(3, (2, 3))#  * 0 + 3
        d = utils.gen_float_tensor(4, (3, 2))#  * 0 + 4
        s = utils.gen_float_tensor(5, (1,))# * 0 + 5
        torch_func = getattr(torch, func)

        a1 = ntnt([a, b])
        if no_grad:
            a2 = ntnt_nograd([b, c])
        else:
            a2 = ntnt([b, c])
        a3 = ntnt([torch_func(a, b),
                   torch_func(b, c)])
        res1 = torch_func(a1, a2)
        if not no_grad:
            res1.sum().backward()
            self.assertIsNotNone(a1.grad)
        if no_grad:
            self.assertIsNone(a2.grad)
        else:
            self.assertIsNotNone(a2.grad)
        self.assertEqual(a3, torch_func(a1, a2))
        self.assertEqual(a3, getattr(a1, func)(a2))
        # a1.detach_()
        # a2.detach_()
        # a3.detach_()
        self.assertEqual(a3, getattr(a1, func + "_")(a2))
        self.assertEqual(a3, a1)

        # Test NT x T
        a1 = ntnt([a, b])
        a2 = c
        a3 = ntnt([torch_func(a, a2),
                   torch_func(b, a2)])

        self.assertEqual(a3, torch_func(a1, a2))
        self.assertEqual(a3, getattr(a1, func)(a2))

        # Test NT x T with broadcasting
        if func not in ["pow", "atan2"]:
            a1 = ntnt([a, b])
            a2 = torch.tensor([1, 2]).reshape(-1, 1, 1)
            a3 = ntnt([torch_func(a, 1),
                       torch_func(b, 2)])
            self.assertEqual(a3, torch_func(a1, a2))
            self.assertEqual(a3, getattr(a1, func)(a2))

        a1 = ntnt([a, d])
        self.assertEqual(ntnt([torch_func(a, s), torch_func(d, s)]),
                         torch_func(a1, s))

        a1 = ntnt([a, b])
        self.assertEqual(ntnt([torch_func(a, c),
                               torch_func(b, c)
                               ]),
                         torch_func(a1, c.reshape(1, 2, 3)))

        result = ntnt([torch_func(c, a),
                       torch_func(c, b)
                       ])
        # if no_grad:
        #     a1.detach_()
        #     result.detach_()
        self.assertEqual(result,
                         torch_func(c.reshape(1, 2, 3), a1))

        # a1 = a1.detach()
        # a3 = a3.detach()
        self.assertEqual(a3, getattr(a1, func + "_")(a2))
        self.assertEqual(a3, a1)

        # The constructor is supposed to copy!
        a1 = c
        a2 = ntnt([a, b])
        a3 = ntnt([torch_func(c, a),
                   torch_func(c, b)])
        # if no_grad:
        #     a2.detach_()
        #     a3.detach_()
        self.assertEqual(a3, torch_func(a1, a2))
        self.assertEqual(a3, getattr(a1, func)(a2))
        # Cannot apply in-place methods to regular Tensors given a NestedTensor as an other
        # TODO: Only sub doesn't adhere to this rule but with irregular behavior
        if func == "add":
            self.assertEqual(c + a + b, getattr(a1, func + "_")(a2))


    return _test_binary


def _gen_test_binary_method(func):
    def _test_binary_method(self):
        a = utils.gen_float_tensor(1, (2, 3))
        b = utils.gen_float_tensor(2, (2, 3))
        c = utils.gen_float_tensor(3, (2, 3))

        # The constructor is supposed to copy!
        a1 = nestedtensor.nested_tensor([a, b])
        a2 = nestedtensor.nested_tensor([b, c])
        a3 = nestedtensor.nested_tensor([getattr(a, "__" + func + "__")(b),
                                         getattr(b, "__" + func + "__")(c)])
        self.assertEqual(a3, getattr(a1, "__" + func + "__")(a2))

        # The constructor is supposed to copy!
        a1 = nestedtensor.nested_tensor([a, b])
        a2 = c
        a3 = nestedtensor.nested_tensor([getattr(a, "__" + func + "__")(a2),
                                         getattr(b, "__" + func + "__")(a2)])
        self.assertEqual(a3, getattr(a1, "__" + func + "__")(a2))

        a1 = c
        a2 = nestedtensor.nested_tensor([a, b])
        a3 = nestedtensor.nested_tensor([getattr(a1, "__" + func + "__")(a),
                                         getattr(a1, "__" + func + "__")(b)])
        self.assertEqual(a3, getattr(a2, "__r" + func + "__")(a1))
    return _test_binary_method


TestUnary = type('TestUnary', (DynamicClassBase,), {})
for func__ in get_unary_functions():
    for nested_dim in range(1, 5):
        avail_devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            avail_devices += [torch.device('cuda')]
        for device in avail_devices:
            setattr(TestUnary, "test_{0}_nested_dim_{1}_{2}".format(
                func__, nested_dim, device), _gen_test_unary(func__, nested_dim, device))

TestBinary = type('TestBinary', (DynamicClassBase,), {})
for func in get_binary_functions():
    no_grad = True
    setattr(TestBinary, "test_{0}".format(func),
            _gen_test_binary(func, no_grad))

# TestBinaryMethod = type('TestBinaryMethod', (DynamicClassBase,), {})
# for func in get_python_binary_arithmetic_operations():
#     # Not implemented yet
#     if func in ['divmod', 'and', 'lshift', 'matmul', 'mod', 'or', 'rshift', 'xor']:
#         continue
#     setattr(TestBinaryMethod, "test_{0}".format(func),
#             _gen_test_binary_method(func))

if __name__ == "__main__":
    unittest.main()
