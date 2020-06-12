import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
import random

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
        self.assertRaisesRegex(RuntimeError, "element 0 of tensors does not require grad and does not have a grad_fn",
                lambda: nt_sum_res.backward())

        self.assertEqual(sum_res, nt_sum_res)
        self.assertIsNone(nt[0].grad)
        self.assertIsNotNone(verification_tensor.grad)

        # nested_tensor constructor
        tensor2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
        nt2 = nestedtensor.nested_tensor([tensor2], requires_grad=True)
        nt_sum_res2 = some_func(nt2)
        nt_sum_res2.backward()
        self.assertEqual(sum_res, nt_sum_res2)
        self.assertIsNone(tensor2.grad)
        self.assertIsNotNone(nt2[0].grad)

    def test_grad_to_tensor_mask(self):
        def some_func(x):
            return torch.sum(x ** 2 + x ** 3)

        nt1 = nestedtensor.as_nested_tensor([torch.tensor([1, 2, 3, 4]),
                                             torch.tensor([1, 2, 3]),
                                             torch.tensor([1, 2])],
                                             dtype=torch.float, requires_grad=True)
        nt_sum_res = some_func(nt1)
        nt_sum_res.backward()

        self.assertEqual(nt1[0].grad, torch.tensor([ 5., 16., 33., 56.]))
        self.assertEqual(nt1[1].grad, torch.tensor([ 5., 16., 33.]))
        self.assertEqual(nt1[2].grad, torch.tensor([ 5., 16.]))


        nt2 = nestedtensor.as_nested_tensor([torch.tensor([1, 2, 3, 4]),
                                             torch.tensor([1, 2, 3]),
                                             torch.tensor([1, 2])],
                                             dtype=torch.float, requires_grad=True)
        # print('nt2')
        # print(nt2)
        # print('nt2.grad')
        # print(nt2.grad)
        tensor, mask = nt2.to_tensor_mask(mask_dim=2)
        sum_res = some_func(tensor)
        sum_res.backward()

        self.assertEqual(sum_res, nt_sum_res)

        print('nt2.grad')
        print(nt2.grad)
        self.assertEqual(nt2[0].grad, torch.tensor([ 5., 16., 33., 56.]))
        self.assertEqual(nt2[1].grad, torch.tensor([ 5., 16., 33.]))
        self.assertEqual(nt2[2].grad, torch.tensor([ 5., 16.]))

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
        self.assertTrue(nt.requires_grad)

        nt_sum_res = some_func(nt)
        nt_sum_res.backward()

        self.assertEqual(total_t_sum, nt_sum_res)
        self.assertEqual(nt[0].grad, torch.tensor([ 5., 16., 33., 56.]))
        self.assertEqual(nt[1].grad, torch.tensor([ 5., 16., 33.]))
        self.assertEqual(nt[2].grad, torch.tensor([ 5., 16.]))


if __name__ == "__main__":
    unittest.main()
