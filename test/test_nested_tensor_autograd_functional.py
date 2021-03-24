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
from frozen_batch_norm_2d import NTFrozenBatchNorm2d
from position_encoding import PositionEmbeddingSine
from joiner import Joiner
from detr_nestedtensor import DETRNestedTensor
from torch import nn


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)
def ntnt_nograd(x): return nestedtensor.nested_tensor(x)


class TestAutogradFunctional(TestCase):
    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
    def test_nn_linear(self):
        def _test(linear):
            inputs = [
                torch.randn(3, 10, requires_grad=True),
                torch.randn(3, 10, requires_grad=True)
            ]

            # most of optional params
            linear = linear()
            tensor_res = []
            for i in range(2):
                t_res = linear(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))
                t_res.sum().backward()
            layer_grad0 = [p.grad for (n, p) in linear.named_parameters()]

            linear.zero_grad()

            nt = ntnt(inputs)
            nt_res = linear(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in linear.named_parameters()]

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(nt.grad[0], inputs[0].grad)
            self.assertEqual(nt.grad[1], inputs[1].grad)

        _test(lambda: torch.nn.Linear(10, 6))

    @unittest.skip("Requires autograd support")
    def test_nn_batch_norm(self):
        def _test(BatchNorm2d, has_grad=True):
            inputs = torch.randn(5, 3, 18, 18, requires_grad=True)

            batch_norm = BatchNorm2d()

            t_res = batch_norm(inputs)
            t_res.sum().backward()
            layer_grad0 = [p.grad for (n, p) in batch_norm.named_parameters()]

            batch_norm.zero_grad()
            nt = ntnt(inputs.unbind())
            nt_res = batch_norm(nt)

            self.assertEqual(ntnt(t_res.unbind()), nt_res)
            if has_grad:
                nt_res.sum().backward()
                layer_grad1 = [p.grad for (
                    n, p) in batch_norm.named_parameters()]
                map(self.assertEqual, zip(layer_grad0, layer_grad1))
                self.assertEqual(nt.grad[0], inputs.grad[0])
                self.assertEqual(nt.grad[1], inputs.grad[1])
            else:
                self.assertRaisesRegex(
                    RuntimeError, "var.dim gradient not implemented yet.", lambda: nt_res.sum().backward())

        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=True), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=True).eval())
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=False), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=False).eval(), False)

        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=False, track_running_stats=False), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=False, track_running_stats=False).eval(), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=False, track_running_stats=True), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=False, track_running_stats=True).eval())
        _test(lambda: torch.nn.BatchNorm2d(3), False)

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
    def test_resnet_bottleneck(self):
        import torchvision

        def _test(Bottleneck, has_grad=True):
            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = Bottleneck()
            print(b)
            x = b(inputs).sum()
            # import torchviz
            # dot = torchviz.make_dot(x)
            # dot.format = 'svg'
            # dot.render('asdf')
            # x.backward()
            # import sys; sys.exit(1)
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True),
                torch.randn(256, 18, 18, requires_grad=True)
            ]
            b = Bottleneck()
            inputs = ntnt(inputs_)
            if has_grad:
                b(inputs).sum().backward()
                # print(list((n, p.grad is None) for (n, p) in b.named_parameters()))

                b.zero_grad()
                b(inputs_[0].unsqueeze(0)).sum().backward()

                b.zero_grad()
                b(inputs_[1].unsqueeze(0)).sum().backward()

                self.assertEqual(inputs_[0].grad, inputs.grad[0])
                self.assertEqual(inputs_[1].grad, inputs.grad[1])
        _test(lambda: torchvision.models.resnet.Bottleneck(256, 64), False)
        _test(lambda: torchvision.models.resnet.Bottleneck(256, 64).eval())

    @unittest.skip("Requires autograd support")
    def test_resnet_classification(self):
        import torchvision

        def _test(FCNHead):
            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = FCNHead()
            print(b)
            # print(b)
            # list(b.children())[3].eval()  # dropout is stochastic otherwise
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
        # _test(lambda: torchvision.models.segmentation.fcn.FCNHead(256, 64))
        _test(lambda: torchvision.models.segmentation.fcn.FCNHead(256, 64).eval())

    @unittest.skip("Requires autograd support")
    def test_backbone(self):
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter

        def _test(FCNHead):
            inputs_ = [
                torch.randn(3, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = FCNHead()
            # print(b)
            # print(b(inputs))
            b(inputs)[0][0].sum().backward()
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0))[0][0].sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]
            inputs = ntnt(inputs_)
            b.zero_grad()
            b(inputs)[0][0].sum().backward()
            # for (n, p) in b.named_parameters():
            #     if p.grad is None:
            #         print(n)
            #         continue
            #     print(n, " is fine")

            b.zero_grad()
            b(inputs_[0].unsqueeze(0))[0][0].sum().backward()

            b.zero_grad()
            b(inputs_[1].unsqueeze(0))[0][0].sum().backward()

            self.assertEqual(inputs_[0].grad, inputs.grad[0])
            self.assertEqual(inputs_[1].grad, inputs.grad[1])
        # Note: It seems expected that layer0 has no gradients.
        return_layers = {"layer1": "0", "layer2": "1",
                         "layer3": "2", "layer4": "3"}
        _test(lambda: Joiner(IntermediateLayerGetter(getattr(torchvision.models, "resnet50")(
            replace_stride_with_dilation=[False, False, False],
            pretrained=True, norm_layer=NTFrozenBatchNorm2d), return_layers),
            PositionEmbeddingSine(128, normalize=True)))

    @unittest.skip("Requires autograd support")
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
        # XXX: This needs a test that actually checks the parameter gradients

    @unittest.skip("Requires autograd support")
    def test_mha_detr(self):
        NDIM = 128
        BSZ = 8
        NHEAD = 8
        RAND_INTS = [(1, 5), (7, 9)]
        MODEL = torch.nn.MultiheadAttention(NDIM, NHEAD).eval()

        src_list = nestedtensor.nested_tensor(
            [torch.randn(NDIM, i, j) for (i, j) in RAND_INTS])
        detr_nt_src = DETRNestedTensor.from_tensor_list(src_list)
        src0, mask = detr_nt_src.decompose()
        src0.requires_grad_()
        src = src0.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        result, _ = MODEL(src, src, src, key_padding_mask=mask,
                          need_weights=False)  # [0].sum().backward()
        mask = (~mask.t().unsqueeze(2)).float()
        result = result * mask
        result_sum = result.sum()
        result_sum.backward()
        grad_sum = src0.grad.sum()

        src = ntnt([t.flatten(1).permute(
            1, 0) for t in src_list])
        result, _ = MODEL(src, src, src, need_weights=False)
        self.assertEqual(result_sum, result.sum())
        result.sum().backward()
        # TODO: The numerical instabilities of summation seem to add up here.
        self.assertEqual(src.grad.sum(), grad_sum, prec=6e-5)

    @unittest.skip("Requires autograd support")
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
        self.assertRaisesRegex(
            RuntimeError, "Cannot squeeze first dimension.", lambda: nt.squeeze(0))
        self.assertRaisesRegex(
            RuntimeError, "Given dimension is either undefined or not a singleton.", lambda: nt.squeeze(2))
        self.assertRaisesRegex(
            RuntimeError, "Given dimension is either undefined or not a singleton.", lambda: nt.squeeze(3))
        self.assertRaises(IndexError, lambda: nt.squeeze(4))
        a = nt.squeeze(1)
        a.sum().backward()
        self.assertEqual(nt.grad, ntnt_nograd(
            [t.reshape(1, 2, 3).mul(0).add(1)]))

        nt = ntnt([[t.reshape(1, 2, 1, 3)]])
        self.assertRaisesRegex(
            RuntimeError, "Cannot squeeze nested dimension.", lambda: nt.squeeze(1))
        # self.assertEqual(nt.squeeze(1), ntnt(
        #     [t.reshape(1, 2, 1, 3)]))
        self.assertEqual(nt.squeeze(
            2), ntnt([[t.reshape(2, 1, 3)]]))
        self.assertEqual(nt.squeeze(
            4), ntnt([[t.reshape(1, 2, 3)]]))

    @unittest.skip("Requires autograd support")
    def test_nn_max_pool2d(self):
        data = [
            [
                torch.randn(3, 500, 600),
                torch.randn(3, 128, 128)
            ],
            [
                torch.randn(3, 500, 600),
                torch.randn(3, 500, 600)
            ],
        ]

        # with optional params
        maxPool2d = torch.nn.MaxPool2d(kernel_size=(
            3, 3), stride=2, padding=(1, 1), dilation=1, ceil_mode=False)
        for inputs in data:
            tensor_res = []
            for i in range(2):
                t_res = maxPool2d(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))

            nt = ntnt(inputs)
            nt_res = maxPool2d(nt)
            self.assertEqual(ntnt(tensor_res), nt_res)

    @unittest.skip("Requires autograd support")
    def test_fzbn2d(self):
        class FrozenBatchNorm2d(torch.nn.Module):
            """
            BatchNorm2d where the batch statistics and the affine parameters are fixed.
            Copy-paste from torchvision.misc.ops with added eps before rqsrt,
            without which any other models than torchvision.models.resnet[18,34,50,101]
            produce nans.
            """

            def __init__(self, n):
                super(FrozenBatchNorm2d, self).__init__()
                self.register_buffer("weight", torch.ones(n))
                self.register_buffer("bias", torch.zeros(n))
                self.register_buffer("running_mean", torch.zeros(n))
                self.register_buffer("running_var", torch.ones(n))

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                num_batches_tracked_key = prefix + 'num_batches_tracked'
                if num_batches_tracked_key in state_dict:
                    del state_dict[num_batches_tracked_key]

                super(FrozenBatchNorm2d, self)._load_from_state_dict(
                    state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs)

            def forward(self, x):
                # move reshapes to the beginning
                # to make it fuser-friendly
                w = self.weight.reshape(-1, 1, 1)
                b = self.bias.reshape(-1, 1, 1)
                rv = self.running_var.reshape(-1, 1, 1)
                rm = self.running_mean.reshape(-1, 1, 1)
                eps = 1e-5
                scale = w * (rv + eps).rsqrt()
                bias = b - rm * scale
                # return (x * scale + bias)
                # return x
                # return (x * scale + bias)
                return x + bias

        b0 = FrozenBatchNorm2d(64)  # .cuda()
        random.seed(1010)
        torch.manual_seed(1310)
        RAND_INTS = [random.randint(100, 300) for _ in range(1)]
        tensors = [torch.rand(64, i, 256, requires_grad=True)
                   for i in RAND_INTS]
        nested_tensor = ntnt(tensors)
        # print(nested_tensor.nested_size())
        s0 = b0(nested_tensor).sum()
        s0.backward()

        b1 = FrozenBatchNorm2d(64)
        s1 = 0
        for t in tensors:
            s1 += b1(t).sum()
        s1.backward()
        self.assertEqual(s0, s1)
        for i in range(len(tensors)):
            self.assertEqual(nested_tensor.grad[i], tensors[i].grad)

        self.assertEqual(len((list(b0.named_parameters()))), 0)
        self.assertEqual(len((list(b1.named_parameters()))), 0)

    @unittest.skip("Requires autograd support")
    def test_layer_norm(self):
        layer_norm = torch.nn.LayerNorm((0,))
        t0 = torch.randn(3)
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self.assertRaisesRegex(RuntimeError,
                               "Cannot normalize across irregular dimension 2", lambda: layer_norm(nt))

        d = torch.nn.Dropout(0.1)
        t0 = torch.randn(864, 256)
        t1 = torch.randn(360, 256)
        ts = [t0, t1, t0, t1]
        nt = ntnt(ts)
        nt2 = ntnt_nograd(ts)
        layer_norm = torch.nn.LayerNorm(256)
        # print(list(layer_norm.named_parameters()))
        # print(nt)
        tt = torch.randn(30, 43, 256, requires_grad=True)
        # print(nt.requires_grad)
        # res = layer_norm(nt)
        res = layer_norm(tt)
        nt = nt + 3
        # print(res.requires_grad)
        res = res * 5
        # print(res)
        # print(res.requires_grad)
        res.sum().backward()
        res = layer_norm(tt + 2)
        res.sum().backward()
        # print(list(layer_norm.named_parameters()))
        # XXX: Need to check weight and bias gradients
        # import sys
        # sys.exit(1)
        t0 = torch.randn(3, 256)
        t1 = torch.randn(2, 256)
        t2 = torch.randn(3, 256)
        ts = [[t0, t1], [t2]]
        result = ntnt(ts)
        map(self.assertEqual, tuple(
            map(lambda x: layer_norm(x), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: layer_norm(x), ts[1])), result[1])

        layer_norm = torch.nn.LayerNorm(3)
        t0 = torch.randn(3, 3, 4)
        t1 = torch.randn(2, 3, 4)
        t2 = torch.randn(3, 3, 4)
        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self.assertRaisesRegex(RuntimeError,
                               "Given normalized_shape=\[3\], expected input with shape \[\*, 3\], but got input of size\[3, 3, 4\]",
                               lambda: layer_norm(nt))

        layer_norm = torch.nn.LayerNorm((3, 2, 4))
        self.assertRaisesRegex(RuntimeError,
                               "Currently only singleton tuples of integers supported for layer_norm.",
                               lambda: layer_norm(nt))

    @unittest.skip("Requires autograd support")
    def test_decoder(self):
        class TransformerDecoderLayer(nn.Module):

            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                         activation="relu", normalize_before=False):
                super().__init__()
                self.self_attn = nestedtensor.nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout)
                self.multihead_attn = nestedtensor.nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout)
                # Implementation of Feedforward model
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)

                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.norm3 = nn.LayerNorm(d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.dropout3 = nn.Dropout(dropout)

                self.activation = torch.nn.functional.relu
                self.normalize_before = normalize_before

            def with_pos_embed(self, tensor, pos):
                return tensor if pos is None else tensor + pos

            def forward(self, tgt, memory,
                        # tgt_mask: Optional[Tensor] = None,
                        # memory_mask: Optional[Tensor] = None,
                        # tgt_key_padding_mask: Optional[Tensor] = None,
                        # memory_key_padding_mask: Optional[Tensor] = None,
                        pos=None, query_pos=None):
                q = k = self.with_pos_embed(tgt, query_pos)
                tgt2 = self.self_attn(q, k, value=tgt,
                                      need_weights=False)[0]
                # tgt = tgt + self.dropout1(tgt2)
                tgt = tgt + tgt2
                tgt = self.norm1(tgt)
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                           key=self.with_pos_embed(
                                               memory, pos),
                                           value=memory,
                                           need_weights=False)[0]
                # tgt = tgt + self.dropout2(tgt2)
                tgt = tgt + tgt2
                tgt = self.norm2(tgt)
                tgt2 = self.linear2(self.dropout(
                    self.activation(self.linear1(tgt))))
                # tgt = tgt + self.dropout3(tgt2)
                tgt = tgt + tgt2
                tgt = self.norm3(tgt)
                # print('tgt.requires_grad')
                # print(tgt.requires_grad)
                return tgt

        d = TransformerDecoderLayer(256, 8)
        d.zero_grad()
        a = d(
            ntnt([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
            ntnt([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
            pos=ntnt([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
            query_pos=ntnt([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
        )
        a.sum().backward()
        # for (n, p) in d.named_parameters():
        #     print(n)
        #     print(p is None)

    def _test_softmax(self, ts, nt):
        fn = F.softmax
        self.assertRaises(RuntimeError, lambda: fn(nt, 0))
        self.assertRaises(RuntimeError, lambda: fn(nt, 1))

        def _map_fn(dim, result):
            result = fn(nt, 2)

            map(self.assertEqual, tuple(
                map(lambda x: fn(x, dim), ts[0])), result[0])
            map(self.assertEqual, tuple(
                map(lambda x: fn(x, dim), ts[1])), result[1])
            s = result.sum()
            # s.backward()
            # ts[0][0].requires_grad_()
            # ts[0][1].requires_grad_()
            # ts[1][0].requires_grad_()
            # map(lambda x: fn(x, dim).sum().backward(), ts[0])
            # map(lambda x: fn(x, dim).sum().backward(), ts[1])
            # map(self.assertEqual, tuple(
            #     map(lambda x: x.grad, ts[0])), nt.grad[0])
            # map(self.assertEqual, tuple(
            #     map(lambda x: x.grad, ts[1])), nt.grad[1])

        for i in range(nt.dim() - nt.nested_dim()):
            _map_fn(i, fn(nt, i + nt.nested_dim()))

    @unittest.skip("Requires autograd support")
    def test_softmax_1(self):
        ts = [[], []]
        nt = ntnt(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Requires autograd support")
    def test_softmax_2(self):
        t0 = torch.randn(3, requires_grad=True)
        t1 = torch.randn(2, requires_grad=True)
        t2 = torch.randn(3, requires_grad=True)
        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Requires autograd support")
    def test_softmax_3(self):
        t0 = torch.randn(3, 2, 1, requires_grad=True)
        t1 = torch.randn(2, 3, 1, requires_grad=True)
        t2 = torch.randn(3, 1, 2, requires_grad=True)
        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Requires autograd support")
    def test_softmax_4(self):
        ts = torch.randn(6, 4, 3, 2, 5, requires_grad=True)
        ts = list(map(lambda x: x.unbind(), ts.unbind()))
        nt = ntnt(ts)
        self._test_softmax(ts, nt)


if __name__ == "__main__":
    unittest.main()
