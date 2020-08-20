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
from torch.nn import functional as F


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)


class TestAutogradFunctional(TestCase):
    def test_nn_conv2d(self):
        def _test(Conv2d):
            inputs = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]

            # most of optional params
            conv2d = Conv2d()
            tensor_res = []
            for i in range(2):
                t_res = conv2d(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))
                t_res.sum().backward()
            layer_grad0 = [p.grad for (n, p) in conv2d.named_parameters()]

            conv2d.zero_grad()

            nt = ntnt(inputs)
            nt_res = conv2d(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in conv2d.named_parameters()]

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(nt.grad[0], inputs[0].grad)
            self.assertEqual(nt.grad[1], inputs[1].grad)

        _test(lambda: torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1), padding=(
            4, 2), padding_mode='zeros', dilation=1, groups=1, bias=True))
        _test(lambda: torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1), padding=(
            4, 2), padding_mode='zeros', dilation=1, groups=1, bias=False))
        _test(lambda: torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1)))

    def test_nn_batch_norm(self):
        def _test(BatchNorm2d):
            inputs = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]

            batch_norm = BatchNorm2d()
            # batch_norm.eval()

            tensor_res = []
            for i in range(2):
                t_res = batch_norm(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))
                t_res.sum().backward()
            print(list(batch_norm.named_parameters()))
            layer_grad0 = [p.grad for (n, p) in batch_norm.named_parameters()]
            print(list(p.sum() for p in layer_grad0))

            batch_norm.zero_grad()
            nt = ntnt(inputs)
            nt_res = batch_norm(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in batch_norm.named_parameters()]
            print(list(p.sum() for p in layer_grad1))

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(nt.grad[0], inputs[0].grad)
            self.assertEqual(nt.grad[1], inputs[1].grad)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True))
        _test(lambda: torch.nn.BatchNorm2d(3))



if __name__ == "__main__":
    unittest.main()
