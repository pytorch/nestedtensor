import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
import random

import utils
from utils import TestCase

class TestNestedTensorAutograd(TestCase):
    def test_basic_grad(self):
        def some_func(x):
            return torch.sum(x ** 2 + x ** 3)
        
        # single tensor case for comparison
        verification_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
        sum_res = some_func(verification_tensor)
        sum_res.backward()

        # as_nested_tensor constructor
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
        nt = nestedtensor.as_nested_tensor([tensor]) 
        nt_sum_res = some_func(nt)
        nt_sum_res.backward()
        self.assertEqual(sum_res, nt_sum_res)
        self.assertEqual(verification_tensor.grad.data, nt[0].grad.data)
        self.assertEqual(verification_tensor.grad.data, tensor.grad.data)

        # nested_tensor constructor
        tensor2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
        nt2 = nestedtensor.nested_tensor([tensor2]) 
        nt_sum_res2 = some_func(nt2)
        nt_sum_res2.backward()
        self.assertEqual(sum_res, nt_sum_res2)
        self.assertEqual(verification_tensor.grad.data, tensor2.grad.data)
        self.assertIsNone(nt2[0].grad)

    def test_grad_to_tensor_mask(self):
        def some_func(x):
            return torch.sum(x ** 2 + x ** 3)

        nt1 = nestedtensor.as_nested_tensor([torch.tensor([1, 2, 3, 4], dtype=torch.float, requires_grad=True),
                                             torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True),
                                             torch.tensor([1, 2], dtype=torch.float, requires_grad=True)])
        nt_sum_res = some_func(nt1)
        nt_sum_res.backward()
        
        nt2 = nestedtensor.as_nested_tensor([torch.tensor([1, 2, 3, 4], dtype=torch.float, requires_grad=True),
                                             torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True),
                                             torch.tensor([1, 2], dtype=torch.float, requires_grad=True)])
        tensor, mask = nt2.to_tensor_mask(mask_dim=2)
        sum_res = some_func(tensor)
        sum_res.backward()
        
        self.assertEqual(sum_res, nt_sum_res)
        self.assertEqual(nt1[0].grad.data, tensor.grad[0].data)
        self.assertEqual(nt1[1].grad.data, tensor.grad[1].data.masked_select(mask[1]))
        self.assertEqual(nt1[2].grad.data, tensor.grad[2].data.masked_select(mask[2]))

        self.assertIsNone(nt2[0].grad)
        self.assertIsNone(nt2[1].grad)
        self.assertIsNone(nt2[2].grad)

    def test_grad_nt_from_tensor_mask(self):
        def some_func(x):
            return torch.sum(x ** 2 + x ** 3)

        t1 = torch.tensor([1., 2., 3., 4.], requires_grad=True)
        t2 = torch.tensor([1., 2., 3.], requires_grad=True)
        t3 = torch.tensor([1., 2.], requires_grad=True)

        res1 = some_func(t1)
        res2 = some_func(t2)
        res3 = some_func(t3)
        total_t_sum = res1 + res2 + res3

        res1.backward()
        res2.backward()
        res3.backward()

        nt_tensor = torch.tensor([[1., 2., 3., 4.],
                                  [1., 2., 3., 0.],
                                  [1., 2., 0., 0.]], requires_grad=True)
        nt_mask = torch.tensor([[ True,  True,  True,  True],
                                [ True,  True,  True, False],
                                [ True,  True, False, False]])

        nt = nestedtensor.nested_tensor_from_tensor_mask(nt_tensor, nt_mask)
        self.assertEqual(True, nt.requires_grad)
        
        nt_sum_res = some_func(nt)
        nt_sum_res.backward()

        self.assertEqual(total_t_sum, nt_sum_res)
        self.assertIsNone(nt[0].grad)
        self.assertIsNone(nt[1].grad)
        self.assertIsNone(nt[2].grad)


if __name__ == "__main__":
    unittest.main()
