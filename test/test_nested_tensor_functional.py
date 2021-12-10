import torch
import nestedtensor
import unittest
from utils_test_case import TestCase
import random
import utils
from torch.nn import functional as F
from detr_nestedtensor import DETRNestedTensor
from torch import nn


def _iter_constructors():
    yield nestedtensor.as_nested_tensor
    yield nestedtensor.nested_tensor


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)


def ntnt_nograd(x, device=None, dtype=None): return nestedtensor.nested_tensor(
    x, requires_grad=False, device=device, dtype=dtype)


class TestFunctional(TestCase):
    def test_nll_loss(self):
        utils.gen_float_tensor(1, (40, 5))
        utils.gen_float_tensor(1, (40,))

    def test_addmm(self):
        torch.rand(5), torch.rand(4, 5)
        nestedtensor.nested_tensor(
            [torch.rand(1, 4), torch.rand(1, 4), torch.rand(4, 4)]
        )

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_add(self):
        nt = ntnt_nograd([torch.randn(4, 2, 5), torch.randn(4, 3, 5)],
                         device=torch.device('cuda'), dtype=torch.half)
        o = torch.randn(1, 4, 1, 1)
        o = o.cuda().half()
        res = nt + o

    def _test_conv2d_dtype(self, dtype, weight, device, shapes,
                           stride=None, padding=None, dilation=None,
                           groups=None):
        if stride is None:
            stride = [1, 1]
        if padding is None:
            padding = [0, 0]
        if dilation is None:
            dilation = [1, 1]
        if groups is None:
            groups = 1

        def _prod(tup):
            r = 1
            for t in tup:
                r = r * t
            return r

        def _test(ts, weight, stride, padding, dilation, groups):
            nt = ntnt_nograd(ts, device=device, dtype=dtype)
            nt_out = torch.conv2d(nt, weight, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups)
            for i, (t, nt_out_i) in enumerate(zip(ts, nt_out.unbind())):
                t_out = torch.conv2d(t.unsqueeze(0), weight,
                                     stride=stride, padding=padding,
                                     dilation=dilation,
                                     groups=groups).squeeze(0)
                self.assertEqual(t_out, nt_out_i)
        ts = []
        for s in shapes:
            ts.append(torch.randn(_prod(s)).reshape(*s).to(device=device, dtype=dtype))
        weight = weight.to(device=device, dtype=dtype)
        _test(ts, weight, stride, padding, dilation, groups)

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_conv2d_1x1_cuda(self):
        shapes = [(2, 2, 3), (2, 4, 2), (2, 2, 2)]
        weight = torch.randn(3*2*1*1).reshape(3, 2, 1, 1)
        self._test_conv2d_dtype(torch.float16, weight, torch.device('cuda'), shapes)
        self._test_conv2d_dtype(torch.float32, weight, torch.device('cuda'), shapes)

    @torch.inference_mode()
    def test_conv2d_1x1_cpu(self):
        shapes = [(2, 2, 3), (2, 4, 2), (2, 2, 2)]
        weight = torch.randn(3*2*1*1).reshape(3, 2, 1, 1)
        # self._test_conv2d_dtype(torch.float16, weight, torch.device('cpu'), shapes)
        self._test_conv2d_dtype(torch.float32, weight, torch.device('cpu'), shapes)

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_conv2d_3x3_cuda(self):
        shapes = [(2, 4, 5), (2, 5, 3), (2, 3, 3)]
        weight = torch.randn(3*2*3*3).reshape(3, 2, 3, 3)
        self._test_conv2d_dtype(torch.float16, weight, torch.device('cuda'), shapes)
        self._test_conv2d_dtype(torch.float32, weight, torch.device('cuda'), shapes)

    @torch.inference_mode()
    def test_conv2d_3x3_cpu(self):
        shapes = [(2, 4, 5), (2, 5, 3), (2, 3, 3)]
        weight = torch.randn(3*2*3*3).reshape(3, 2, 3, 3)
        # self._test_conv2d_dtype(torch.float16, weight, torch.device('cpu'), shapes)
        self._test_conv2d_dtype(torch.float32, weight, torch.device('cpu'), shapes)

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_conv2d_3x3_resnext_common_cuda(self):
        shapes = [(32, 4, 5), (32, 5, 3), (32, 3, 3)]
        weight = torch.randn(32*1*3*3).reshape(32, 1, 3, 3)
        for dtype in [torch.float16, torch.float32]:
            stride = [1, 1]  # default
            padding = [1, 1]
            dilation = [1, 1]  # default
            groups = 32
            self._test_conv2d_dtype(dtype, weight, torch.device('cuda'),
                                    shapes, stride=stride, padding=padding,
                                    dilation=dilation, groups=groups)

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_conv2d_3x3_resnext_input_cuda(self):
        shapes = [(4, 3, 2), (4, 3, 3), (4, 2, 3)]
        weight = torch.randn(5, 4, 2, 2)
        for dtype in [torch.float16, torch.float32]:
            stride = [1, 1]
            padding = [1, 1]
            dilation = [1, 1]
            groups = 1
            self._test_conv2d_dtype(dtype, weight, torch.device('cuda'),
                                    shapes, stride=stride, padding=padding,
                                    dilation=dilation, groups=groups)

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

        # nt_cont = relu(nt)
        # self.assertEqual(True, nt_cont.is_contiguous())

    @torch.inference_mode()
    def test_nn_embedding(self):
        inputs = [torch.randint(100, (L,)) for L in torch.randint(5, 50, (8,))]
        x = nestedtensor.nested_tensor(inputs, dtype=torch.int64)
        emb = torch.nn.Embedding(100, 8)
        y = emb(x)
        for i, inp in enumerate(inputs):
            self.assertEqual(emb(inp), y[i])

    @torch.inference_mode()
    def test_nn_embedding_bag(self):

        def run_test(EmbeddingBag, inputs):
            x = nestedtensor.nested_tensor(inputs, dtype=torch.int64)
            torch.manual_seed(0)
            emb = EmbeddingBag()
            y = emb(x)
            s = y.sum()
            # s.backward()
            input_tensor = torch.cat(inputs).contiguous()
            input_offset = [0]
            for inp in inputs[:-1]:
                input_offset.append(len(inp) + input_offset[-1])
            input_offset = torch.tensor(input_offset)
            torch.manual_seed(0)
            emb_t = EmbeddingBag()
            y_t = emb_t(input_tensor, input_offset)
            s_t = y_t.sum()
            # s_t.backward()
            for yi, y_ti in zip(y.unbind(), y_t.unbind()):
                self.assertEqual(yi, y_ti)
            self.assertEqual(s, s_t)
            # self.assertEqual(emb.weight.grad, emb_t.weight.grad)

        run_test(lambda: torch.nn.EmbeddingBag(100, 8), [
                 torch.randint(100, (5,)), torch.randint(100, (5,))])
        run_test(lambda: torch.nn.EmbeddingBag(100, 8), [
                 torch.randint(100, (L,)) for L in torch.randint(3, 7, (5,))])
        run_test(lambda: torch.nn.EmbeddingBag(100, 8, sparse=True), [
                 torch.randint(100, (5,)), torch.randint(100, (5,))])
        run_test(lambda: torch.nn.EmbeddingBag(100, 8, sparse=True), [
                 torch.randint(100, (L,)) for L in torch.randint(3, 7, (5,))])

    @torch.inference_mode()
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

    @unittest.skip("Not implemented")
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
        expected_nt = ntnt_nograd([expected_t])

        t_clone = orig_t1.clone()
        torch.nn.functional.relu_(t_clone)
        self.assertEqual(t_clone, expected_t)

        t_clone = orig_t1.clone()
        nt1 = ntnt_nograd([t_clone])
        torch.nn.functional.relu_(nt1)
        self.assertEqual(nt1, expected_nt)
        self.assertEqual(t_clone, orig_t1)

        t_clone = orig_t1.clone()
        nt1 = nestedtensor.as_nested_tensor([t_clone])
        torch.nn.functional.relu_(nt1)
        self.assertEqual(nt1, expected_nt)
        self.assertNotEqual(t_clone, expected_t)

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

        input_nt = nestedtensor.nested_tensor(inputs)
        target_nt = nestedtensor.nested_tensor(targets, dtype=torch.int64)
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

        nt = ntnt_nograd(inputs)
        nt_res = torch.nn.functional.dropout(nt)
        self.assertEqual(ntnt_nograd(tensor_res).size(), nt_res.size())

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

            # Currently only supporting nested dim 1.
            # nt1 = constructor(
            #     [[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            # nt2 = constructor(
            #     [[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            # nt1.copy_(nt2)
            # self.assertEqual(nt1, nt2)

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_unsqueeze(self):
        for constructor in _iter_constructors():
            t = torch.randn(2, 3)

            # Currently only supporting nested dim 1.
            # nt = constructor([[t.reshape(2, 3)]])
            # self.assertEqual(nt.unsqueeze(
            #     0), constructor([[[t.reshape(2, 3)]]]))
            # self.assertEqual(nt.unsqueeze(
            #     1), constructor([[[t.reshape(2, 3)]]]))
            # self.assertEqual(nt.unsqueeze(
            #     2), constructor([[t.reshape(1, 2, 3)]]))
            # self.assertEqual(nt.unsqueeze(
            #     3), constructor([[t.reshape(2, 1, 3)]]))
            # self.assertEqual(nt.unsqueeze(
            #     4), constructor([[t.reshape(2, 3, 1)]]))

            # Currently only supporting nested dim 1.
            # t0 = t.reshape(3, 2)
            # t1 = t
            # t2 = torch.randn(4, 5)
            # nt = constructor([[t0, t1], [t2]])
            # self.assertEqual(nt.unsqueeze(0), constructor([[[t0, t1], [t2]]]))
            # self.assertEqual(nt.unsqueeze(
            #     1), constructor([[[t0, t1]], [[t2]]]))
            # self.assertEqual(nt.unsqueeze(2), constructor(
            #     [[t0.reshape(1, 3, 2), t1.reshape(1, 2, 3)], [t2.reshape(1, 4, 5)]]))
            # self.assertEqual(nt.unsqueeze(3), constructor(
            #     [[t0.reshape(3, 1, 2), t1.reshape(2, 1, 3)], [t2.reshape(4, 1, 5)]]))
            # self.assertEqual(nt.unsqueeze(4), constructor(
            #     [[t0.reshape(3, 2, 1), t1.reshape(2, 3, 1)], [t2.reshape(4, 5, 1)]]))

            t = torch.randn(2, 3)
            nt = constructor([t])
            self.assertEqual(nt.unsqueeze(0), constructor([[t]]))
            self.assertEqual(nt.unsqueeze(
                1), constructor([t.reshape(1, 2, 3)]))
            self.assertEqual(nt.unsqueeze(
                2), constructor([t.reshape(2, 1, 3)]))
            self.assertEqual(nt.unsqueeze(
                3), constructor([t.reshape(2, 3, 1)]))
            self.assertRaises(IndexError, lambda: nt.unsqueeze(4))

    @torch.inference_mode()
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
            # Currently only supporting nested dim 1.
            # c = constructor([[t21, t22], [t22, t21]])
            # result2 = torch.matmul(c, t1)
            # self.assertEqual(result2[0][0], torch.matmul(t21, t1))
            # self.assertEqual(result2[0][1], torch.matmul(t22, t1))
            # self.assertEqual(result2[1][0], torch.matmul(t22, t1))
            # self.assertEqual(result2[1][1], torch.matmul(t21, t1))

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_transpose(self):
        t0 = torch.randn(3, 3, 4)
        t1 = torch.randn(2, 4, 3)
        t2 = torch.randn(3, 3, 2)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        self.assertRaisesRegex(RuntimeError, "Transposition of nested dimensions is not implemented yet.",
                               lambda: nt.transpose(0, 2))
        self.assertRaisesRegex(RuntimeError, "Transposition of nested dimensions is not implemented yet.",
                               lambda: nt.transpose(1, 3))
        self.assertRaisesRegex(RuntimeError, "Transposition of nested dimensions is not implemented yet.",
                               lambda: nt.transpose(0, 1))
        self.assertEqual(nt.transpose(2, 3), nt.transpose(3, 2))
        t = torch.randn(2, 3, 2, 4, 1)
        t_t = t.transpose(2, 3)
        nt = nestedtensor.nested_tensor(
            list(map(lambda x: x.unbind(), t.unbind())))
        nt_t = nestedtensor.nested_tensor(
            list(map(lambda x: x.unbind(), t_t.unbind())))
        self.assertEqual(t_t, nt_t.to_tensor())

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_flatten(self):
        t0 = torch.randn(3, 3, 4)
        t1 = torch.randn(2, 4, 3)
        t2 = torch.randn(3, 3, 2)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        self.assertRaisesRegex(RuntimeError, "Cannot flatten nested dimension 0",
                               lambda: nt.flatten(0))
        self.assertRaisesRegex(RuntimeError, "Cannot flatten nested dimension 1",
                               lambda: nt.flatten(2, 1))
        result = nt.flatten(2)
        map(self.assertEqual, tuple(
            map(lambda x: x.flatten(), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: x.flatten(), ts[1])), result[1])

        result = nt.flatten(3, 4)
        map(self.assertEqual, tuple(
            map(lambda x: x.flatten(1, 2), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: x.flatten(1, 2), ts[1])), result[1])

        ts = torch.randn(3, 2, 4, 5, 3)
        ts_r = ts.flatten(3, 4)
        ts = list(map(lambda x: x.unbind(), ts.unbind()))
        ts_r = list(map(lambda x: x.unbind(), ts_r.unbind()))
        ts = nestedtensor.nested_tensor(ts).flatten(3, 4)
        ts_r = nestedtensor.nested_tensor(ts_r)
        map(self.assertEqual, zip(ts[0].unbind(), ts_r[0].unbind()))
        map(self.assertEqual, zip(ts[1].unbind(), ts_r[1].unbind()))

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_reshape(self):
        t0 = torch.randn(3, 3)
        t1 = torch.randn(2, 3)
        t2 = torch.randn(3, 3)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)
        self.assertRaisesRegex(RuntimeError, "Reshape cannot be exclusive to nested dimensions.",
                               lambda: nt.reshape(0, -1))
        self.assertRaisesRegex(RuntimeError, "Cannot reshape explicitly along irregular dimension 1. Please use -1 as a placeholder.",
                               lambda: nt.reshape(-1, 1, 2, 3))
        result = nt.reshape(-1, -1, 3, -1)
        map(self.assertEqual, tuple(
            map(lambda x: x.reshape(3, -1), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: x.reshape(3, -1), ts[1])), result[1])

        result = nt.reshape(-1, -1, 1, 1, 3, -1)
        map(self.assertEqual, tuple(
            map(lambda x: x.reshape(1, 1, 3, -1), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: x.reshape(1, 1, 3, -1), ts[1])), result[1])

        result = nt.reshape(-1, -1, 1, 1, 3, -1)
        map(self.assertEqual, tuple(
            map(lambda x: x.reshape(1, 1, 3, -1), ts[0])), result[0])
        map(self.assertEqual, tuple(
            map(lambda x: x.reshape(1, 1, 3, -1), ts[1])), result[1])

        ts = torch.randn(3, 2, 4, 5, 3)
        ts_r = ts.reshape(3, 2, 5, 3, 4)
        ts = list(map(lambda x: x.unbind(), ts.unbind()))
        ts_r = list(map(lambda x: x.unbind(), ts_r.unbind()))
        ts = nestedtensor.nested_tensor(ts)
        ts_r = nestedtensor.nested_tensor(ts_r)
        map(self.assertEqual, zip(ts[0].unbind(), ts_r[0].unbind()))
        map(self.assertEqual, zip(ts[1].unbind(), ts_r[1].unbind()))

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

        for i in range(nt.dim() - nt.nested_dim()):
            _map_fn(i, fn(nt, i + nt.nested_dim()))

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_softmax_1(self):
        ts = [[], []]
        nt = ntnt_nograd(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_softmax_2(self):
        t0 = torch.randn(3)
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        ts = [[t0, t1], [t2]]
        nt = ntnt_nograd(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_softmax_3(self):
        t0 = torch.randn(3, 2, 1)
        t1 = torch.randn(2, 3, 1)
        t2 = torch.randn(3, 1, 2)
        ts = [[t0, t1], [t2]]
        nt = ntnt_nograd(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Currently only supporting nested dim 1.")
    def test_softmax_4(self):
        ts = torch.randn(6, 4, 3, 2, 5)
        ts = list(map(lambda x: x.unbind(), ts.unbind()))
        nt = ntnt_nograd(ts)
        self._test_softmax(ts, nt)

    @torch.inference_mode()
    def test_mha(self):
        embed_dim = 2
        num_heads = 2
        torch.manual_seed(1010)
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads)
        query = torch.randn(3, 1, embed_dim, requires_grad=True)
        key = torch.randn(2, 1, embed_dim, requires_grad=True)
        value = torch.randn(2, 1, embed_dim, requires_grad=True)
        attn_output, _ = mha(query, key, value)
        nt_mha = torch.nn.MultiheadAttention(embed_dim, num_heads)
        nt_mha.in_proj_weight = mha.in_proj_weight
        nt_mha.in_proj_bias = mha.in_proj_bias
        nt_mha.out_proj.weight = mha.out_proj.weight
        nt_mha.out_proj.bias = mha.out_proj.bias
        query_nt = ntnt_nograd([query.squeeze(1)])
        key_nt = ntnt_nograd([key.squeeze(1)])
        value_nt = ntnt_nograd([value.squeeze(1)])
        nt_attn_output, _ = nt_mha(
            query_nt, key_nt, value_nt, need_weights=False)
        self.assertEqual(attn_output.squeeze(1), nt_attn_output[0])

    @torch.inference_mode()
    def test_mha_detr(self):
        NDIM = 128
        BSZ = 8
        NHEAD = 8
        RAND_INTS = [(1, 5), (7, 9)]
        MODEL = torch.nn.MultiheadAttention(NDIM, NHEAD).eval()

        src_list = ntnt_nograd(
            [torch.randn(NDIM, i, j) for (i, j) in RAND_INTS])
        detr_nt_src = DETRNestedTensor.from_tensor_list(src_list)
        src0, mask = detr_nt_src.decompose()
        src0.requires_grad_()
        src = src0.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        result, _ = MODEL(src, src, src, key_padding_mask=mask,
                          need_weights=False)  # [0].sum().backward()
        mask = (~mask.t().unsqueeze(2)).float()
        result0 = result * mask
        # result_sum = result.sum()

        src = ntnt_nograd([t.flatten(1).permute(
            1, 0) for t in src_list])
        result1, _ = MODEL(src, src, src, need_weights=False)
        self.assertEqual(result0.sum(0).sum(0), result1.sum(1).sum(0))

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_mha_detr_cuda(self):
        NDIM = 128
        BSZ = 8
        NHEAD = 8
        RAND_INTS = [(1, 5), (7, 9)]
        MODEL = torch.nn.MultiheadAttention(NDIM, NHEAD).cuda().eval()

        src_list = [torch.randn(NDIM, i, j) for (i, j) in RAND_INTS]
        detr_nt_src = DETRNestedTensor.from_tensor_list(src_list)
        src0, mask = detr_nt_src.decompose()
        src = src0.flatten(2).permute(2, 0, 1).cuda()
        mask = mask.flatten(1).cuda()
        result, _ = MODEL(src, src, src, key_padding_mask=mask,
                          need_weights=False)  # [0].sum().backward()
        mask = (~mask.t().unsqueeze(2)).float()
        result0 = result * mask
        # result_sum = result.sum()

        src = ntnt_nograd([t.flatten(1).permute(
            1, 0) for t in src_list], device=torch.device('cuda'))
        result1, _ = MODEL(src, src, src, need_weights=False)
        self.assertEqual(result0.sum(0).sum(0), result1.sum(1).sum(0))

    def test_squeeze(self):
        t = torch.randn(2, 3)
        result = ntnt_nograd([t])

        # Currently only supporting nested dim 1.
        # nt = ntnt_nograd([[t.reshape(1, 2, 1, 3)]])
        # # self.assertEqual(nt.squeeze(), result)
        # self.assertRaises(RuntimeError, lambda: nt.squeeze())
        # nt.squeeze_()
        # self.assertEqual(nt, result)

        nt = ntnt_nograd([t.reshape(2, 3)])
        # self.assertEqual(nt.squeeze(), result)
        self.assertRaises(RuntimeError, lambda: nt.squeeze())
        nt.squeeze_()
        self.assertEqual(nt, result)

        # Currently only supporting nested dim 1.
        # nt = ntnt_nograd([[t.reshape(2, 3)]])
        # # self.assertEqual(nt.squeeze(), result)
        # self.assertRaises(RuntimeError, lambda: nt.squeeze())
        # nt.squeeze_()
        # self.assertEqual(nt, result)

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

        # Currently only supporting nested dim 1.
        # nt = ntnt_nograd([[[t.reshape(1, 2, 3)]]])
        # # self.assertEqual(nt.squeeze(), result)
        # self.assertRaises(RuntimeError, lambda: nt.squeeze())
        # nt.squeeze_()
        # self.assertEqual(nt, result)

        # result = ntnt([t])
        # nt = ntnt([t.reshape(1, 2, 3)])
        # self.assertEqual(nt.squeeze(1), result)
        # self.assertRaisesRegex(
        #     RuntimeError, "Cannot squeeze first dimension.", lambda: nt.squeeze(0))
        # self.assertRaisesRegex(
        #     RuntimeError, "Given dimension is either undefined or not a singleton.", lambda: nt.squeeze(2))
        # self.assertRaisesRegex(
        #     RuntimeError, "Given dimension is either undefined or not a singleton.", lambda: nt.squeeze(3))
        # self.assertRaises(IndexError, lambda: nt.squeeze(4))
        # a = nt.squeeze(1)
        # a.sum().backward()
        # self.assertEqual(nt.grad, ntnt_nograd(
        #     [t.reshape(1, 2, 3).mul(0).add(1)]))

        # nt = ntnt([[t.reshape(1, 2, 1, 3)]])
        # self.assertRaisesRegex(
        #     RuntimeError, "Cannot squeeze nested dimension.", lambda: nt.squeeze(1))
        # # self.assertEqual(nt.squeeze(1), ntnt(
        # #     [t.reshape(1, 2, 1, 3)]))
        # self.assertEqual(nt.squeeze(
        #     2), ntnt([[t.reshape(2, 1, 3)]]))
        # self.assertEqual(nt.squeeze(
        #     4), ntnt([[t.reshape(1, 2, 3)]]))

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

            nt = ntnt_nograd(inputs)
            nt_res = maxPool2d(nt)
            self.assertEqual(ntnt_nograd(tensor_res), nt_res)

    @unittest.skip("Currently broken")
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
                print("1")
                w = self.weight.reshape(-1, 1, 1)
                print("2")
                b = self.bias.reshape(-1, 1, 1)
                print("3")
                rv = self.running_var.reshape(-1, 1, 1)
                print("4")
                rm = self.running_mean.reshape(-1, 1, 1)
                print("5")
                eps = 1e-5
                print("6")
                scale = w * (rv + eps).rsqrt()
                print("7")
                bias = b - rm * scale
                print("8")
                # return (x * scale + bias)
                # return x
                # return (x * scale + bias)
                res = x + bias
                print("9")
                return res

        b0 = FrozenBatchNorm2d(64)  # .cuda()
        random.seed(1010)
        torch.manual_seed(1310)
        RAND_INTS = [random.randint(100, 300) for _ in range(1)]
        tensors = [torch.rand(64, i, 256, requires_grad=False)
                   for i in RAND_INTS]
        # RAND_INTS = [random.randint(1, 1) for _ in range(1)]
        # tensors = [torch.rand(1, i, 2, requires_grad=True)
        #            for i in RAND_INTS]
        nested_tensor = ntnt_nograd(tensors)
        # print(nested_tensor.nested_size())
        s00 = b0(nested_tensor)
        print("s00")
        print(s00.requires_grad)
        s0 = s00.sum()
        # s0.backward()

        b1 = FrozenBatchNorm2d(64)
        s1 = 0
        for t in tensors:
            s1 += b1(t).sum()
        # s1.backward()
        self.assertEqual(s0, s1)
        # for i in range(len(tensors)):
        #     self.assertEqual(nested_tensor.grad[i], tensors[i].grad)

        self.assertEqual(len((list(b0.named_parameters()))), 0)
        self.assertEqual(len((list(b1.named_parameters()))), 0)

    @torch.inference_mode()
    def test_layer_norm(self):
        def _test(device):
            # Currently only supporting nested dim 1.
            # layer_norm = torch.nn.LayerNorm((0,)).to(device)
            # t0 = torch.randn(3)
            # t1 = torch.randn(2)
            # t2 = torch.randn(3)
            # ts = [[t0, t1], [t2]]
            # nt = ntnt_nograd(ts, device=device)
            # self.assertRaisesRegex(RuntimeError,
            #                        "Cannot normalize across irregular dimension 2", lambda: layer_norm(nt))

            t0 = utils.gen_float_tensor(1, (2, 32)).to(device)
            t1 = utils.gen_float_tensor(2, (2, 32)).to(device)
            ts = [t0, t1, t0, t1]
            nt = ntnt_nograd(ts, device=device)
            layer_norm = torch.nn.LayerNorm(32).to(device)
            nt_result = layer_norm(nt)
            for i in range(len(ts)):
                self.assertEqual(nt_result[i], layer_norm(
                    ts[i].reshape(1, -1, 32).squeeze(0)))

            layer_norm = torch.nn.LayerNorm(16).to(device)
            tt = utils.gen_float_tensor(1, (3, 23, 16)).to(device)
            res = layer_norm(tt)
            nt = nt + 3
            res = res * 5
            res = layer_norm(tt + 2)
            t0 = utils.gen_float_tensor(1, (3, 16)).to(device)
            t1 = utils.gen_float_tensor(2, (2, 16)).to(device)
            t2 = utils.gen_float_tensor(3, (3, 16)).to(device)

            # Currently only supporting nested dim 1.
            # ts = [[t0, t1], [t2]]
            # result = ntnt_nograd(ts, device=device)
            # layer_norm(ts[0][0])
            # map(self.assertEqual, tuple(
            #     map(lambda x: layer_norm(x), ts[0])), result[0])
            # map(self.assertEqual, tuple(
            #     map(lambda x: layer_norm(x), ts[1])), result[1])

            # layer_norm = torch.nn.LayerNorm(3).to(device)
            # t0 = torch.randn(3, 3, 4)
            # t1 = torch.randn(2, 3, 4)
            # t2 = torch.randn(3, 3, 4)
            # ts = [[t0, t1], [t2]]
            # nt = ntnt_nograd(ts, device=device)
            # self.assertRaisesRegex(RuntimeError,
            #                        "Normalized shape \[3\] does not match the size of the last dimension \(4\) of input.",
            #                        lambda: layer_norm(nt))

            # layer_norm = torch.nn.LayerNorm((3, 2, 4)).to(device)
            # self.assertRaisesRegex(RuntimeError,
            #                        "Currently only singleton tuples of integers supported for layer_norm.",
            #                        lambda: layer_norm(nt))
        _test(torch.device('cpu'))
        if torch.cuda.is_available():
            _test(torch.device('cuda'))

    @torch.inference_mode()
    def test_decoder(self):
        class TransformerDecoderLayer(nn.Module):

            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                         activation="relu", normalize_before=False):
                super().__init__()
                self.self_attn = torch.nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout)
                self.multihead_attn = torch.nn.MultiheadAttention(
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
            ntnt_nograd([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
            ntnt_nograd([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
            pos=ntnt_nograd([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
            query_pos=ntnt_nograd([
                torch.randn(864, 256),
                torch.randn(360, 256)]),
        )
        # a.sum().backward()
        # for (n, p) in d.named_parameters():
        #     print(n)
        #     print(p is None)

    @torch.inference_mode()
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires cuda")
    def test_effective_transformer_mha(self):

        def test(num_heads, batch_size, seq_len_, head_size, embedding_dim,
                 use_arange=False):
            assert num_heads * head_size == embedding_dim
            import random
            inputs = []
            k = 0
            seq_len = 0
            seq_lens = []
            for _ in range(batch_size):
                i = random.randint(1, seq_len_)
                seq_len = max(i, seq_len)
                seq_lens.append(i)
                if use_arange:
                    inputs.append(torch.arange(
                        i * embedding_dim).reshape(i, embedding_dim))
                else:
                    inputs.append(torch.randn(i, embedding_dim))
            input_nt = nestedtensor.nested_tensor(
                inputs, device=torch.device('cuda'), dtype=torch.float)

            input_batch, input_mask = input_nt.to_tensor_mask(mask_dim=2)

            mha = torch.nn.MultiheadAttention(embedding_dim, num_heads)
            if use_arange:
                in_proj_weight_test = torch.arange(mha.in_proj_weight.numel()).reshape(
                    mha.in_proj_weight.shape).to(torch.float)
                mha.in_proj_weight.copy_(in_proj_weight_test)
            in_proj_weight = mha.in_proj_weight.clone().cuda()

            in_proj_bias = mha.in_proj_bias.clone().cuda()

            if use_arange:
                out_proj_weight_test = torch.arange(mha.out_proj.weight.numel()).reshape(
                    mha.out_proj.weight.shape).to(torch.float)
                mha.out_proj.weight.copy_(
                    out_proj_weight_test)
            out_proj_weight = mha.out_proj.weight.clone().cuda()

            import time
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.time()
            scaling = float(head_size ** -0.5)
            for _ in range(5):
                result_nt = torch.ops.nestedtensor.bt_min_mha(num_heads,
                                                              head_size,
                                                              0.5,
                                                              False,
                                                              input_nt,
                                                              input_nt,
                                                              input_nt,
                                                              in_proj_weight,
                                                              in_proj_bias,
                                                              scaling,
                                                              out_proj_weight,
                                                              in_proj_bias)

            torch.cuda.synchronize()
            t1 = time.time()
            a = t1 - t0

            mha = mha.cuda()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                attn_output, _ = mha(input_nt, input_nt, input_nt)

            torch.cuda.synchronize()
            t1 = time.time()
            b = t1 - t0

            self.assertEqual(result_nt, attn_output)

            torch.cuda.synchronize()
            input_batch = input_batch.transpose(0, 1)
            not_input_mask = torch.logical_not(input_mask)
            torch.cuda.synchronize()
            t0 = time.time()
            # print(input_batch.size())
            for _ in range(5):
                attn_output, _ = mha(
                    input_batch,
                    input_batch,
                    input_batch,
                    key_padding_mask=not_input_mask)


            torch.cuda.synchronize()
            t1 = time.time()
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output * torch.logical_not(not_input_mask.unsqueeze(-1))
            self.assertEqual(result_nt.to_padded_tensor(padding=0), attn_output)
            c = t1 - t0
            print("bt: ", a, "\tnt: ", b, "\tdense: ", c, "\tdense/bt: ", c/a)

        # test(1, 1, 1, 4, 4, use_arange=True)
        # test(1, 1, 2, 2, 2, use_arange=True)
        # test(1, 2, 2, 1, 1, use_arange=True)
        # test(1, 4, 3, 2, 2, use_arange=True)
        test(2, 1, 2, 1, 2)
        test(1, 3, 5, 4, 4)
        test(2, 3, 5, 2, 4)
        test(2, 1, 2, 2, 4)
        test(2, 1, 2, 2, 4)
        test(2, 3, 5, 2, 4)
        test(1, 3, 5, 4, 4)
        test(8, 8, 50, 16, 128)
        test(16, 64, 50, 16, 256)
        test(16, 128, 50, 16, 256)
        test(16, 256, 50, 16, 256)
        test(4,  256, 50, 256, 1024)
        test(16, 256, 50, 64, 1024)

    @torch.inference_mode()
    def test_relu(self):
        nt = ntnt_nograd([torch.randn(2, 3), torch.randn(3, 2)])
        n1 = torch.nn.ReLU(inplace=False)
        out1 = n1(nt)
        n2 = torch.nn.ReLU(inplace=True)
        out2 = n2(nt)
        self.assertEqual(out1, out2)
        self.assertEqual(out1, nt)


if __name__ == "__main__":
    unittest.main()
