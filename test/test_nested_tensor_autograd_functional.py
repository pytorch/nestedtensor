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
def ntnt_nograd(x): return nestedtensor.nested_tensor(x)


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
            layer_grad0 = [p.grad for (n, p) in batch_norm.named_parameters()]

            batch_norm.zero_grad()
            nt = ntnt(inputs)
            nt_res = batch_norm(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in batch_norm.named_parameters()]

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(nt.grad[0], inputs[0].grad)
            self.assertEqual(nt.grad[1], inputs[1].grad)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=True))
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
        #                                    affine=True, track_running_stats=True).eval())
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
        #                                    momentum=0.1, affine=False, track_running_stats=False))
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
        #                                    affine=False, track_running_stats=False).eval())
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
        #                                    momentum=0.1, affine=True, track_running_stats=False))
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
        #                                    affine=True, track_running_stats=False).eval())
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
        #                                    momentum=0.1, affine=False, track_running_stats=True))
        # _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
        #                                    affine=False, track_running_stats=True).eval())
        # _test(lambda: torch.nn.BatchNorm2d(3))

    def test_nn_relu(self):
        inputs = [
            torch.randn(3, 500, 600, requires_grad=True),
            torch.randn(3, 128, 128, requires_grad=True)
        ]

        relu = torch.nn.ReLU()
        relu_ = torch.nn.ReLU(inplace=True)
        tensor_res = []
        for i in range(2):
            t_res = relu(inputs[i].unsqueeze(0).contiguous())
            t_res = relu_(t_res)
            tensor_res.append(t_res.squeeze(0))
            tensor_res[i].sum().backward()
        layer_grad0 = [p.grad for (n, p) in relu.named_parameters()]

        nt = ntnt(inputs)
        nt_res = relu(nt)
        nt_res = relu_(nt_res)
        nt_res.sum().backward()
        layer_grad1 = [p.grad for (n, p) in relu.named_parameters()]

        self.assertEqual(ntnt(tensor_res), nt_res)
        map(self.assertEqual, zip(layer_grad0, layer_grad1))
        self.assertEqual(inputs[0].grad, nt.grad[0])
        self.assertEqual(inputs[1].grad, nt.grad[1])

    def test_add(self):
        inputs0_ = [
            torch.randn(5, 6, requires_grad=True),
            torch.randn(1, 1, requires_grad=True)
        ]
        inputs1_ = [
            torch.randn(5, 6, requires_grad=True),
            torch.randn(1, 1, requires_grad=True)
        ]
        inputs0 = ntnt(inputs0_)
        inputs1 = ntnt(inputs1_)
        output = inputs0 + inputs1
        output += inputs0
        output.sum().backward()
        self.assertEqual(inputs0.grad.sum(),
                         inputs1.grad.sum() + inputs1.grad.sum())

    def test_resnet_bottleneck(self):
        import torchvision

        def _test(Bottleneck):
            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = Bottleneck()
            b(inputs).sum().backward()
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True),
                torch.randn(256, 18, 18, requires_grad=True)
            ]
            inputs = ntnt(inputs_)
            b.zero_grad()
            b(inputs).sum().backward()

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()

            b.zero_grad()
            b(inputs_[1].unsqueeze(0)).sum().backward()

            self.assertEqual(inputs_[0].grad, inputs.grad[0])
            self.assertEqual(inputs_[1].grad, inputs.grad[1])
        _test(lambda: torchvision.models.resnet.Bottleneck(256, 64))
        _test(lambda: torchvision.models.resnet.Bottleneck(256, 64).eval())

    def test_resnet_classification(self):
        import torchvision

        def _test(FCNHead):
            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = FCNHead()
            list(b.children())[3].eval()  # dropout is stochastic otherwise
            b(inputs).sum().backward()
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True),
                torch.randn(256, 18, 18, requires_grad=True)
            ]
            inputs = ntnt(inputs_)
            b.zero_grad()
            b(inputs).sum().backward()

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()

            b.zero_grad()
            b(inputs_[1].unsqueeze(0)).sum().backward()

            self.assertEqual(inputs_[0].grad, inputs.grad[0])
            self.assertEqual(inputs_[1].grad, inputs.grad[1])
        _test(lambda: torchvision.models.segmentation.fcn.FCNHead(256, 64))
        _test(lambda: torchvision.models.segmentation.fcn.FCNHead(256, 64).eval())

    def test_mha(self):
        embed_dim = 2
        num_heads = 2
        torch.manual_seed(1010)
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads)
        query = torch.randn(3, 1, embed_dim, requires_grad=True)
        key = torch.randn(2, 1, embed_dim, requires_grad=True)
        value = torch.randn(2, 1, embed_dim, requires_grad=True)
        attn_output, _ = mha(query, key, value)
        nt_mha = nestedtensor.nn.MultiheadAttention(embed_dim, num_heads)
        nt_mha.in_proj_weight = mha.in_proj_weight
        nt_mha.in_proj_bias = mha.in_proj_bias
        nt_mha.out_proj.weight = mha.out_proj.weight
        nt_mha.out_proj.bias = mha.out_proj.bias
        query_nt = ntnt([query.squeeze(1)])
        key_nt = ntnt([key.squeeze(1)])
        value_nt = ntnt([value.squeeze(1)])
        nt_attn_output, _ = nt_mha(
            query_nt, key_nt, value_nt, need_weights=False)
        # nt_attn_output.sum().backward()
        # For regular tensors the batch dimension is along dimension 1
        scalar1 = attn_output.sum()
        scalar2 = nt_attn_output.sum()
        scalar1.backward()
        scalar2.backward()
        self.assertEqual(attn_output.squeeze(1), nt_attn_output[0])

    def test_squeeze(self):
        t = torch.randn(2, 3)
        result = ntnt_nograd([t])

        nt = ntnt_nograd([[t.reshape(1, 2, 1, 3)]])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        nt = ntnt_nograd([t.reshape(2, 3)])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        nt = ntnt_nograd([[t.reshape(2, 3)]])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        nt = ntnt_nograd([t.reshape(1, 2, 3)])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        nt = ntnt_nograd([t.reshape(1, 2, 1, 3, 1)])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        nt = ntnt_nograd([[[t.reshape(1, 2, 3)]]])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        result = ntnt([t])
        nt = ntnt([t.reshape(1, 2, 3)])
        self.assertEqual(nt.squeeze(1), result)
        self.assertRaisesRegex(RuntimeError, "Cannot squeeze first dimension.", lambda: nt.squeeze(0))
        self.assertRaisesRegex(RuntimeError, "Given dimension is either undefined or not a singleton.", lambda: nt.squeeze(2))
        self.assertRaisesRegex(RuntimeError, "Given dimension is either undefined or not a singleton.", lambda: nt.squeeze(3))
        self.assertRaises(IndexError, lambda: nt.squeeze(4))
        a = nt.squeeze(1)
        a.sum().backward()
        self.assertEqual(nt.grad, ntnt_nograd([t.reshape(1, 2, 3).mul(0).add(1)]))

        nt = ntnt([[t.reshape(1, 2, 1, 3)]])
        self.assertRaisesRegex(RuntimeError, "Cannot squeeze nested dimension.", lambda: nt.squeeze(1))
        # self.assertEqual(nt.squeeze(1), ntnt(
        #     [t.reshape(1, 2, 1, 3)]))
        self.assertEqual(nt.squeeze(
            2), ntnt([[t.reshape(2, 1, 3)]]))
        self.assertEqual(nt.squeeze(
            4), ntnt([[t.reshape(1, 2, 3)]]))


if __name__ == "__main__":
    unittest.main()
