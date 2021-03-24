import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from utils import TestCase
import random


# TODO: Test unbind, test grad and backward


class TestNestedTensorBuffer(TestCase):
    @unittest.skip("Requires autograd support")
    def test_grad(self):
        nt = nestedtensor.nested_tensor([torch.rand(1, 2)])
        nt.requires_grad_(True)
        a = nt.unbind()[0]
        c = nt.sum()
        c.backward()
        # TODO: Should this have a gradient or not?
        # What if nt was constructed with as_nested_tensor vs. nested_tensor
        # When calling unbind on a torch.Tensor it doesn't have a grad,
        # because it is not a leaf variable. So, if we call unbind
        # on a NestedTensor to retrieve one of its constiuents, should
        # that be a leaf variable or a view?
        # Further, if the NestedTensor has a buffer, the constiuents are
        # views of that buffer, so that means unbind() needs to return views
        # in either case.

        # When I set requires_grad_ for a NestedTensor and this NestedTensors becomes
        # a leaf in an autograd graph it'll have a .grad field. If I call unbind on
        # this NestedTensor I should get a list of views. However, if constructed
        # with as_nested_tensor I'll get a list of Tensors, i.e. the Tensors used
        # to actually build the NestedTensor, which are then all leaf variables
        # (since requires_grad_ is forwarded to its constiuents since .grad()
        # on the constiuents is used to construct NestedTensor.grad)

        # TODO: Re-enable under autograd
        # self.assertIsNotNone(a.grad)
        # nt_grad = nt.grad
        # # Unbinding the gradient is legitimate for further processing.
        # self.assertIsNotNone(nt_grad.unbind()[0])

    # TODO
    @unittest.skip("Requires autograd support")
    def test_detach(self):
        pass


if __name__ == "__main__":
    unittest.main()
