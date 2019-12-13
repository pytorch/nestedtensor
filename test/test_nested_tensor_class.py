import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from unittest import TestCase
import random

import utils

class TestNestedTensor(TestCase):

    # def test_nested_constructor(self):
    #     num_nested_tensor = 3
    #     # TODO: Shouldn't be constructable
    #     nested_tensors = [utils.gen_nested_tensor(i, i, 3)
    #                       for i in range(1, num_nested_tensor)]
    #     nested_tensor = nestedtensor.nested_tensor(nested_tensors)

    def test_constructor(self):
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(utils.gen_float_tensor(i, (i + 1, 128, 128)))
        nested_tensor = nestedtensor.nested_tensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertTrue((tensors[i] != nested_tensor.unbind()[i]).all())
        self.assertRaises(
            ValueError, lambda: nestedtensor.nested_tensor(torch.tensor([3.0])))
        self.assertRaises(ValueError, lambda: nestedtensor.nested_tensor(
            nestedtensor.nested_tensor([torch.tensor([3.0])])))
        self.assertRaises(ValueError, lambda: nestedtensor.nested_tensor(
            [torch.tensor([2.0]), nestedtensor.nested_tensor([torch.tensor([3.0])])]))
        self.assertRaises(ValueError, lambda: nestedtensor.nested_tensor(4.0))

    def test_default_constructor(self):
        self.assertRaises(TypeError, lambda: nestedtensor.nested_tensor())
        # nested_dim is 1 and dim is 1 too.
        default_nested_tensor = nestedtensor.nested_tensor([])
        default_tensor = torch.tensor([])
        self.assertEqual(default_nested_tensor.nested_dim(), 1)
        self.assertEqual(default_nested_tensor.nested_size(), nestedtensor.NestedSize([]))
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(default_nested_tensor.requires_grad,
                         default_tensor.requires_grad)
        self.assertEqual(default_nested_tensor.is_pinned(),
                         default_tensor.is_pinned())

    def test_scalar_constructor(self):
        dim_one_nested_tensor = nestedtensor.nested_tensor([1.0])
        self.assertEqual(dim_one_nested_tensor.dim(), 1)
        self.assertEqual(dim_one_nested_tensor.nested_dim(), 1)

    def test_nested_size(self):
        a = nestedtensor.nested_tensor(
            [torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
        na = nestedtensor.NestedSize([[1, 2], [2, 3], [4, 5]])
        self.assertEqual(a.nested_size(), na)
        values = [torch.rand(1, 2) for i in range(10)]
        values = [values[1:i] for i in range(2, 10)]
        nt = nestedtensor.nested_tensor(values)
        nts = nt.nested_size(1)
        lens = tuple(map(len, values))
        self.assertTrue(nts == lens)

    def test_len(self):
        a = nestedtensor.nested_tensor([torch.tensor([1, 2]),
                                 torch.tensor([3, 4]),
                                 torch.tensor([5, 6]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 4)
        a = nestedtensor.nested_tensor([torch.tensor([1, 2]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 2)
        a = nestedtensor.nested_tensor([torch.tensor([1, 2])])
        self.assertEqual(len(a), 1)

    def test_equal(self):
        a1 = nestedtensor.nested_tensor([torch.tensor([1, 2]),
                                  torch.tensor([7, 8])])
        a2 = nestedtensor.nested_tensor([torch.tensor([1, 2]),
                                  torch.tensor([7, 8])])
        a3 = nestedtensor.nested_tensor([torch.tensor([3, 4]),
                                  torch.tensor([5, 6])])
        # Just exercising them until we have __bool__, all() etc.
        self.assertTrue((a1 == a2).all())
        self.assertTrue((a1 != a3).all())
        self.assertTrue(not (a1 != a2).any())
        self.assertTrue(not (a1 == a3).any())

    def test_nested_dim(self):
        nt = nestedtensor.nested_tensor([torch.tensor(3)])
        for i in range(2, 5):
            nt = utils.gen_nested_tensor(i, i, 3)
            self.assertEqual(nt.nested_dim(), i)

    def test_unbind(self):
        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nt = nestedtensor.nested_tensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue((a == a1).all())
        self.assertTrue((b == b1).all())

        a = utils.gen_float_tensor(1, (2, 3)).add_(1)
        nt = nestedtensor.nested_tensor([a])
        self.assertTrue((a == nt.unbind()[0]).all())

    def test_size(self):
        a1 = nestedtensor.nested_tensor([[torch.rand(1, 8),
                                   torch.rand(3, 8)],
                                  [torch.rand(7, 8)]])
        a2 = nestedtensor.nested_tensor([torch.rand(1, 2),
                                  torch.rand(1, 8)])
        a3 = nestedtensor.nested_tensor([torch.rand(3, 4),
                                  torch.rand(5, 4)])
        self.assertTrue(a1.size() == (2, None, None, 8))
        self.assertTrue(a2.size() == (2, 1, None))
        self.assertTrue(a3.size() == (2, None, 4))

class TestContiguous(TestCase):
    def test_contiguous(self):
        for i in range(1, 10):
            # data = gen_nested_list(1, 2, 3, size_low=1, size_high=3)
            data = [[torch.rand(1, 2), torch.rand(3, 4)], [torch.rand(5, 6)]]
            nt = nestedtensor.nested_tensor(data)
            self.assertTrue(nt.is_contiguous())
            # buf = nt.flatten()
            self.assertTrue((nt == nt).all())
            a = nt + nt
        nt.cos_()
        nt.cos()

if __name__ == "__main__":
    unittest.main()
