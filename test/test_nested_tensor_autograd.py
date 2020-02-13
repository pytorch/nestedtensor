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

    def test_grad(self):
        a1 = nestedtensor.nested_tensor(
            [
                [torch.tensor([[1.0, 2.0], [2.0, 3.0]]), torch.tensor([[3.0, 4.0]])],
                [torch.tensor([[4.0]])]
            ])
        a2 = nestedtensor.nested_tensor(
            [
                [torch.tensor([[2.5, 1.5], [2.2, 3.3]]), torch.tensor([[0.3, 0.4]])],
                [torch.tensor([[2.4]])]
            ])
        a1.requires_grad_()
        a2.requires_grad_()
        scalars = (a1 + a2)
        res = scalars.sum(2).sum(2)


if __name__ == "__main__":
    unittest.main()
