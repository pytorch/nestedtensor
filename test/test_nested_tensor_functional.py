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


def _iter_constructors():
    yield nestedtensor.as_nested_tensor
    yield nestedtensor.nested_tensor


class TestFunctional(TestCase):
    def test_nll_loss(self):
        utils.gen_float_tensor(1, (40, 5))
        utils.gen_float_tensor(1, (40,))

    def test_addmm(self):
        torch.rand(5), torch.rand(4, 5)
        nestedtensor.nested_tensor(
            [torch.rand(1, 4), torch.rand(1, 4), torch.rand(4, 4)]
        )

    def test_contiguousity(self):
        initial_t = torch.rand(2, 5, 10, 15)
        self.assertEqual(True, initial_t.is_contiguous())

        non_contiguous_1 = initial_t.select(1, 0)
        non_contiguous_2 = initial_t.select(1, 0)
        self.assertEqual(False, non_contiguous_1.is_contiguous())

        relu = torch.nn.ReLU()
        t_cont = relu(non_contiguous_1)
        self.assertEqual(True, t_cont.is_contiguous())

        nt = nestedtensor.nested_tensor([non_contiguous_1, non_contiguous_2])
        self.assertEqual(True, nt.is_contiguous())

        nt_cont = relu(nt)
        self.assertEqual(True, nt_cont.is_contiguous())

    def test_nn_conv2d(self):
        inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

        # most of optional params
        conv2d = torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1), padding=(
            4, 2), padding_mode='zeros', dilation=1, groups=1, bias=True)
        tensor_res = []
        for i in range(2):
            t_res = conv2d(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = conv2d(nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

        # some of optional params
        conv2d = torch.nn.Conv2d(3, 33, kernel_size=3, bias=False)
        tensor_res = []
        for i in range(2):
            t_res = conv2d(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = conv2d(nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_nn_functional_conv2d(self):
        tensor1 = torch.rand(3, 128, 128)
        tensor2 = torch.rand(3, 300, 400)
        inputs = [tensor1, tensor2]
        weight = torch.rand(3, 3, 7, 7)

        # no optional params
        tensor_res = [torch.nn.functional.conv2d(
            t.unsqueeze(0), weight).squeeze(0) for t in inputs]
        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = [t for t in torch.nn.functional.conv2d(
                nt, weight).unbind()]
            self.assertEqual(nt_res, tensor_res)

        # optional params with no bias
        tensor_res = [torch.nn.functional.conv2d(t.unsqueeze(
            0), weight, None, 2, 3, 1, 1).squeeze(0) for t in inputs]
        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = [t for t in torch.nn.functional.conv2d(
                nt, weight, None, 2, 3, 1, 1).unbind()]
            self.assertEqual(nt_res, tensor_res)

        # optional params with bias
        bias = torch.rand(3)
        tensor_res = [torch.nn.functional.conv2d(t.unsqueeze(
            0), weight, bias, (2, 2), (3, 3), (1, 1), 1).squeeze(0) for t in inputs]
        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = [t for t in torch.nn.functional.conv2d(
                nt, weight, bias, (2, 2), (3, 3), (1, 1), 1).unbind()]
            self.assertEqual(nt_res, tensor_res)

    def test_nn_batch_norm(self):
        inputs = [
            torch.tensor([[[-0.5000]], [[0.5000]]]),
            torch.tensor([[[-1.0000, 1.0000], [-0.2500, -0.5000]],
                          [[0.2500, 0.5000], [1.5000, -1.5000]]])
        ]

        batch_norm = torch.nn.BatchNorm2d(2, 1e-05, 0.1)
        batch_norm.eval()

        tensor_res = []
        for i in range(2):
            t_res = batch_norm(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = batch_norm(nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_nn_functional_batch_norm(self):
        inputs = [
            torch.tensor([[[-0.5000]], [[0.5000]]]),
            torch.tensor([[[-1.0000, 1.0000], [-0.2500, -0.5000]],
                          [[0.2500, 0.5000], [1.5000, -1.5000]]])
        ]

        tensor_res = []
        running_mean = torch.rand(2)
        running_var = torch.rand(2)
        for i in range(2):
            t_res = torch.nn.functional.batch_norm(
                inputs[i].unsqueeze(0).contiguous(), running_mean, running_var)
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = torch.nn.functional.batch_norm(
                nt, running_mean, running_var)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

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

            for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
                nt_res = maxPool2d(nt)
                self.assertEqual(
                    nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_nn_functional_max_pool2d(self):
        inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.max_pool2d(inputs[i].unsqueeze(0).contiguous(), kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = torch.nn.functional.max_pool2d(nt, kernel_size=(3, 3), stride=(
                2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_functional_relu_(self):
        orig_t1 = torch.tensor([-2, -1, 0, 1, 2])
        expected_t = torch.tensor([0, 0, 0, 1, 2])
        expected_nt = nestedtensor.nested_tensor([expected_t])

        t_clone = orig_t1.clone()
        torch.nn.functional.relu_(t_clone)
        self.assertEqual(t_clone, expected_t)

        t_clone = orig_t1.clone()
        nt1 = nestedtensor.nested_tensor([t_clone])
        torch.nn.functional.relu_(nt1)
        self.assertEqual(nt1, expected_nt)
        self.assertEqual(t_clone, orig_t1)

        t_clone = orig_t1.clone()
        nt1 = nestedtensor.as_nested_tensor([t_clone])
        torch.nn.functional.relu_(nt1)
        self.assertEqual(nt1, expected_nt)
        self.assertNotEqual(t_clone, expected_t)

    def test_nn_relu(self):
        inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

        relu = torch.nn.ReLU()

        tensor_res = []
        for i in range(2):
            t_res = relu(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = relu(nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_nn_functional_relu(self):
        inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.relu(
                inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = torch.nn.functional.relu(nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_nn_functional_cross_entropy(self):
        inputs = [
            torch.randn(3, 300, 300),
            torch.randn(3, 400, 400)
        ]

        targets = [
            torch.randint(1, (300, 300), dtype=torch.int64),
            torch.randint(1, (400, 400), dtype=torch.int64)
        ]

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.cross_entropy(
                inputs[i].unsqueeze(0).contiguous(), targets[i].unsqueeze(0))
            tensor_res.append(t_res.squeeze(0))

        for input_nt, target_nt in [(nestedtensor.nested_tensor(inputs), nestedtensor.nested_tensor(targets)),
                                    (nestedtensor.as_nested_tensor(inputs), nestedtensor.as_nested_tensor(targets))]:
            nt_res = torch.nn.functional.cross_entropy(input_nt, target_nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_nn_dropout(self):
        inputs = [
            torch.randn(3, 128, 128),
            torch.randn(3, 300, 400)
        ]

        dropout = torch.nn.Dropout(p=0.2)
        tensor_res = []
        for i in range(2):
            t_res = dropout(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = dropout(nt)
            self.assertEqual(nestedtensor.nested_tensor(
                tensor_res).size(), nt_res.size())

    def test_nn_functional_dropout(self):
        inputs = [
            torch.randn(3, 128, 128),
            torch.randn(3, 300, 400)
        ]

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.dropout(
                inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = torch.nn.functional.dropout(nt)
            torch.nn.functional.dropout(nt, inplace=True)
            self.assertEqual(nestedtensor.nested_tensor(
                tensor_res).size(), nt_res.size())

    def test_nn_functional_interpolate(self):
        inputs = [
            torch.randn(3, 200, 300),
            torch.randn(3, 300, 400)
        ]

        # no optional params
        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.interpolate(
                inputs[i].unsqueeze(0).contiguous(), 200)
            tensor_res.append(t_res.squeeze(0))

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = torch.nn.functional.interpolate(nt, 200)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

        # tuple/int size and optional mode
        for size in [(200, 200), 100]:
            tensor_res = []
            for i in range(2):
                t_res = torch.nn.functional.interpolate(inputs[i].unsqueeze(
                    0).contiguous(), size, mode='bilinear', align_corners=True)
                tensor_res.append(t_res.squeeze(0))

            for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
                nt_res = torch.nn.functional.interpolate(
                    nt, size, mode='bilinear', align_corners=True)
                self.assertEqual(
                    nestedtensor.nested_tensor(tensor_res), nt_res)

        # special NT case - list of sizes
        size = ((100, 100), (200, 250), )
        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = torch.nn.functional.interpolate(
                nt, size, mode='bilinear', align_corners=True)
            self.assertEqual(nt_res.nested_size(2), (100, 200))
            self.assertEqual(nt_res.nested_size(3), (100, 250))

        # scale_factor instead of a size
        for scale_factor in [(2.2, 2.2), 1.1]:
            tensor_res = []
            for i in range(2):
                t_res = torch.nn.functional.interpolate(
                    inputs[i].unsqueeze(0).contiguous(), scale_factor=scale_factor)
                tensor_res.append(t_res.squeeze(0))

            for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
                nt_res = torch.nn.functional.interpolate(
                    nt, scale_factor=scale_factor)
                self.assertEqual(
                    nestedtensor.nested_tensor(tensor_res), nt_res)

        # check errors
        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.interpolate(
                nt, size=(100, 100), scale_factor=(1, 1)))

    def test_copy_(self):
        for constructor in _iter_constructors():
            nt1 = constructor([])
            nt2 = constructor([])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

            nt1 = constructor([torch.randn(1, 2, 3)])
            nt2 = constructor([torch.randn(1, 2, 3)])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

            nt1 = constructor([torch.randn(1, 2, 3), torch.randn(2, 1, 3)])
            nt2 = constructor([torch.randn(1, 2, 3), torch.randn(2, 1, 3)])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

            nt1 = constructor(
                [[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            nt2 = constructor(
                [[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            nt1.copy_(nt2)
            self.assertEqual(nt1, nt2)

    def test_squeeze(self):
        for constructor in _iter_constructors():
            t = torch.randn(2, 3)
            result = constructor([t])

            nt = constructor([[t.reshape(1, 2, 1, 3)]])
            self.assertEqual(nt.squeeze(), result)
            nt.squeeze_()
            self.assertEqual(nt, result)

            nt = constructor([t.reshape(2, 3)])
            self.assertEqual(nt.squeeze(), result)
            nt.squeeze_()
            self.assertEqual(nt, result)

            nt = constructor([[t.reshape(2, 3)]])
            self.assertEqual(nt.squeeze(), result)
            nt.squeeze_()
            self.assertEqual(nt, result)

            nt = constructor([t.reshape(1, 2, 3)])
            self.assertEqual(nt.squeeze(), result)
            nt.squeeze_()
            self.assertEqual(nt, result)

            nt = constructor([t.reshape(1, 2, 1, 3, 1)])
            self.assertEqual(nt.squeeze(), result)
            nt.squeeze_()
            self.assertEqual(nt, result)

            nt = constructor([[[t.reshape(1, 2, 3)]]])
            self.assertEqual(nt.squeeze(), result)
            nt.squeeze_()
            self.assertEqual(nt, result)

            nt = constructor([t.reshape(1, 2, 3)])
            self.assertEqual(nt.squeeze(1), result)
            self.assertRaises(RuntimeError, lambda: nt.squeeze(0))
            self.assertRaises(RuntimeError, lambda: nt.squeeze(2))
            self.assertRaises(RuntimeError, lambda: nt.squeeze(3))
            self.assertRaises(IndexError, lambda: nt.squeeze(4))

            nt = constructor([[t.reshape(1, 2, 1, 3)]])
            self.assertEqual(nt.squeeze(1), constructor(
                [t.reshape(1, 2, 1, 3)]))
            self.assertEqual(nt.squeeze(
                2), constructor([[t.reshape(2, 1, 3)]]))
            self.assertEqual(nt.squeeze(
                4), constructor([[t.reshape(1, 2, 3)]]))

    def test_matmul(self):
        for constructor in _iter_constructors():
            t1 = torch.randn(2, 3)
            a = constructor([t1, t1])
            t21 = torch.randn(3, 2)
            t22 = torch.randn(3, 2)
            b = constructor([t21, t22])
            result = torch.matmul(a, b)
            result1 = torch.matmul(a, t22)
            self.assertEqual(result[1], result1[0])
            self.assertEqual(result[1], result1[1])
            c = constructor([[t21, t22], [t22, t21]])
            result2 = torch.matmul(c, t1)
            self.assertEqual(result2[0][0], torch.matmul(t21, t1))
            self.assertEqual(result2[0][1], torch.matmul(t22, t1))
            self.assertEqual(result2[1][0], torch.matmul(t22, t1))
            self.assertEqual(result2[1][1], torch.matmul(t21, t1))

    def test_mha(self):
        embed_dim = 2
        num_heads = 2
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads)
        query = torch.randn(3, 1, embed_dim)
        key = torch.randn(2, 1, embed_dim)
        value = torch.randn(2, 1, embed_dim)
        attn_output, _ = mha(query, key, value)
        nt_mha = nestedtensor.nn.MultiheadAttention(embed_dim, num_heads)
        nt_mha.in_proj_weight = mha.in_proj_weight
        nt_mha.in_proj_bias = mha.in_proj_bias
        nt_mha.out_proj.weight = mha.out_proj.weight
        nt_mha.out_proj.bias = mha.out_proj.bias
        query_nt = nestedtensor.nested_tensor([query.squeeze(1)])
        key_nt = nestedtensor.nested_tensor([key.squeeze(1)])
        value_nt = nestedtensor.nested_tensor([value.squeeze(1)])
        nt_attn_output, _ = nt_mha(
            query_nt, key_nt, value_nt, need_weights=False)
        # For regular tensors the batch dimension is along dimension 1
        self.assertEqual(attn_output.squeeze(1), nt_attn_output[0])

    def test_layer_norm(self):
        layer_norm = torch.nn.LayerNorm((0,))
        t0 = torch.randn(3)
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        self.assertRaisesRegex(RuntimeError,
                               "Cannot normalize across irregular dimension 2", lambda: layer_norm(nt))

        layer_norm = torch.nn.LayerNorm((3,))
        t0 = torch.randn(3, 3)
        t1 = torch.randn(2, 3)
        t2 = torch.randn(3, 3)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        result = F.layer_norm(nt, (3,))
        map(self.assertEqual, tuple(
            map(lambda x: layer_norm(x), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: layer_norm(x), ts[1])), result[1])

        t0 = torch.randn(3, 3, 4)
        t1 = torch.randn(2, 3, 4)
        t2 = torch.randn(3, 3, 4)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        self.assertRaisesRegex(RuntimeError,
                               "Given normalized_shape=\[3\], expected input with shape \[\*, 3\], but got input of size\[3, 3, 4\]",
                               lambda: layer_norm(nt))

        layer_norm = torch.nn.LayerNorm((3, 2, 4))
        self.assertRaisesRegex(RuntimeError,
                               "Currently only singleton tuples of integers supported for layer_norm.",
                               lambda: layer_norm(nt))


def make_generic_map_tests(fn):
    def _test(self, ts, nt):
        self.assertRaises(RuntimeError, lambda: fn(nt, 0))
        self.assertRaises(RuntimeError, lambda: fn(nt, 1))

        def _map_fn(dim, result):
            result = fn(nt, 2)
            map(self.assertEqual, tuple(
                map(lambda x: fn(x, dim), ts[0])), result[0])
            map(self.assertEqual, tuple(
                map(lambda x: fn(x, dim), ts[1])), result[1])

        for i in range(nt.dim() - nt.nested_dim()):
            _map_fn(i, fn(nt, i + nt.nested_dim()))

    def _test_1(self):
        ts = [[], []]
        nt = nestedtensor.nested_tensor(ts)
        _test(self, ts, nt)

    def _test_2(self):
        t0 = torch.randn(3)
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        _test(self, ts, nt)

    def _test_3(self):
        t0 = torch.randn(3, 2, 1)
        t1 = torch.randn(2, 3, 1)
        t2 = torch.randn(3, 1, 2)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        _test(self, ts, nt)

    def _test_4(self):
        ts = torch.randn(6, 4, 3, 2, 5)
        ts = list(map(lambda x: x.unbind(), ts.unbind()))
        nt = nestedtensor.nested_tensor(ts)
        _test(self, ts, nt)

    return [_test_1, _test_2, _test_3, _test_4]


for i, test in enumerate(make_generic_map_tests(F.softmax)):
    setattr(TestFunctional, 'test_softmax_' + str(i), test)


if __name__ == "__main__":
    unittest.main()
