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
        _test(lambda: torch.nn.Conv2d(
            3, 33, kernel_size=(1, 1), stride=(1, 1), bias=False))

    def test_nn_batch_norm(self):
        def _test(BatchNorm2d):
            inputs = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]

            batch_norm = BatchNorm2d()
            batch_norm.eval()

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
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=True))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=True).eval())
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=False, track_running_stats=False))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=False, track_running_stats=False).eval())
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=False))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=False).eval())
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=False, track_running_stats=True))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=False, track_running_stats=True).eval())
        _test(lambda: torch.nn.BatchNorm2d(3))

    def test_nn_relu(self):
        def _test(ReLU):
            inputs = [
                torch.randn(3, 500, 600, requires_grad=True),
                torch.randn(3, 128, 128, requires_grad=True)
            ]

            relu = ReLU()
            tensor_res = []
            for i in range(2):
                t_res = relu(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))
                tensor_res[i].sum().backward()
            print(list(relu.named_parameters()))
            layer_grad0 = [p.grad for (n, p) in relu.named_parameters()]
            print(list(p.sum() for p in layer_grad0))

            nt = ntnt(inputs)
            nt_res = relu(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in relu.named_parameters()]
            print(list(p.sum() for p in layer_grad1))

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(inputs[0].grad, nt.grad[0])
            self.assertEqual(inputs[1].grad, nt.grad[1])
        _test(lambda: torch.nn.ReLU())
        # _test(lambda: torch.nn.ReLU(inplace=True))

    def test_add(self):
        inputs0_ = [
            torch.randn(3, 50, 60, requires_grad=True),
            torch.randn(3, 18, 18, requires_grad=True)
        ]
        inputs1_ = [
            torch.randn(3, 50, 60, requires_grad=True),
            torch.randn(3, 18, 18, requires_grad=True)
        ]
        inputs0 = ntnt(inputs0_)
        inputs1 = ntnt(inputs1_)
        print("A")
        output = inputs0 + inputs1
        print("B")
        output.sum().backward()
        print(inputs0.grad.sum())
        print(inputs1.grad.sum())

        inputs0 = ntnt(inputs0_)
        inputs1 = ntnt(inputs1_)
        print("A")
        inputs0 += inputs1
        print("B")
        output.sum().backward()
        print(inputs0.grad.sum())
        print(inputs1.grad.sum())

    def test_integration(self):
        import torchvision
        inputs_ = [
            torch.randn(256, 50, 60, requires_grad=True),
            torch.randn(256, 18, 18, requires_grad=True)
        ]
        inputs = ntnt(inputs_)
        b = torchvision.models.resnet.Bottleneck(256, 64)# .eval()
        # a = torchvision.models.segmentation.fcn.FCNHead(256, 21).eval()
        print(b)
        # print(a)

        # a(b(inputs)).sum().backward()
        b(inputs).sum().backward()
        print(inputs.grad.sum())
        print(list((n, p.grad.sum()) for (n, p) in b.named_parameters()))
        # print(list(p.grad.sum() for (n, p) in a.named_parameters()))

        b = torchvision.models.resnet.Bottleneck(256, 64)# .eval()
        # a = torchvision.models.segmentation.fcn.FCNHead(256, 21).eval()
        print(b)
        # print(a)

        # a(b(inputs_[0].unsqueeze(0))).sum().backward()
        b(inputs_[0].unsqueeze(0)).sum().backward()
        print(inputs_[0].grad.sum())
        print(list((n, p.grad.sum()) for (n, p) in b.named_parameters()))
        # print(list(p.grad.sum() for (n, p) in a.named_parameters()))

    def test_batch_norm_conv2d(self):
        def _test(BatchNorm2d, Conv2d):
            inputs = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]

            nt = ntnt(inputs)

            batch_norm = BatchNorm2d()
            batch_norm.eval()
            nt_res = batch_norm(nt)

            conv2d = Conv2d()
            nt_res = conv2d(nt_res)

            batch_norm0 = BatchNorm2d()
            batch_norm0.eval()
            nt_res = batch_norm0(nt_res)

            nt_res += ident

            relu = torch.nn.ReLU(inplace=True)
            nt_res = relu(nt_res)

            nt_res.sum().backward()
            print(list(batch_norm.named_parameters()))
            print(list((n, p.grad.sum()) for (n, p) in conv2d.named_parameters()))
            print(list(batch_norm0.named_parameters()))

            # batch_norm = BatchNorm2d()
            # batch_norm.eval()
            # nt_res = batch_norm(inputs[0].unsqueeze(0))
            # conv2d = Conv2d()
            # nt_res = conv2d(nt_res)
            # nt_res.sum().backward()
            # print(list(batch_norm.named_parameters()))
            # print(list((n, p.grad.sum()) for (n, p) in conv2d.named_parameters()))
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=True),
        lambda: torch.nn.Conv2d(3, 3, kernel_size=3, stride=(2, 1), padding=(
            4, 2), padding_mode='zeros', dilation=1, groups=1, bias=True))




if __name__ == "__main__":
    unittest.main()
