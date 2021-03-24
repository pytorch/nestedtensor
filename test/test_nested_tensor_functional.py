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


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)


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

        # nt_cont = relu(nt)
        # self.assertEqual(True, nt_cont.is_contiguous())

    @unittest.skip("Requires autograd support")
    def test_nn_embedding(self):
        inputs = [torch.randint(100, (L,)) for L in torch.randint(5, 50, (8,))]
        x = nestedtensor.nested_tensor(inputs, dtype=torch.int64)
        emb = torch.nn.Embedding(100, 8)
        y = emb(x)
        for i, inp in enumerate(inputs):
            self.assertEqual(emb(inp), y[i])

    @unittest.skip("Requires autograd support")
    def test_nn_embedding_bag(self):

        def run_test(EmbeddingBag, inputs):
            x = nestedtensor.nested_tensor(inputs, dtype=torch.int64)
            torch.manual_seed(0)
            emb = EmbeddingBag()
            y = emb(x)
            s = y.sum()
            s.backward()
            input_tensor = torch.cat(inputs).contiguous()
            input_offset = [0]
            for inp in inputs[:-1]:
                input_offset.append(len(inp) + input_offset[-1])
            input_offset = torch.tensor(input_offset)
            torch.manual_seed(0)
            emb_t = EmbeddingBag()
            y_t = emb_t(input_tensor, input_offset)
            s_t = y_t.sum()
            s_t.backward()
            for yi, y_ti in zip(y.unbind(), y_t.unbind()):
                self.assertEqual(yi, y_ti)
            self.assertEqual(s, s_t)
            self.assertEqual(emb.weight.grad, emb_t.weight.grad)

        run_test(lambda: torch.nn.EmbeddingBag(100, 8), [torch.randint(100, (5,)), torch.randint(100, (5,))])
        run_test(lambda: torch.nn.EmbeddingBag(100, 8), [torch.randint(100, (L,)) for L in torch.randint(3, 7, (5,))])
        run_test(lambda: torch.nn.EmbeddingBag(100, 8, sparse=True), [torch.randint(100, (5,)), torch.randint(100, (5,))])
        run_test(lambda: torch.nn.EmbeddingBag(100, 8, sparse=True), [torch.randint(100, (L,)) for L in torch.randint(3, 7, (5,))])


    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

        nt = ntnt(inputs)
        nt_res = torch.nn.functional.dropout(nt)
        self.assertEqual(ntnt(tensor_res).size(), nt_res.size())

    @unittest.skip("Requires autograd support")
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

    def test_unsqueeze(self):
        for constructor in _iter_constructors():
            t = torch.randn(2, 3)

            nt = constructor([[t.reshape(2, 3)]])
            self.assertEqual(nt.unsqueeze(
                0), constructor([[[t.reshape(2, 3)]]]))
            self.assertEqual(nt.unsqueeze(
                1), constructor([[[t.reshape(2, 3)]]]))
            self.assertEqual(nt.unsqueeze(
                2), constructor([[t.reshape(1, 2, 3)]]))
            self.assertEqual(nt.unsqueeze(
                3), constructor([[t.reshape(2, 1, 3)]]))
            self.assertEqual(nt.unsqueeze(
                4), constructor([[t.reshape(2, 3, 1)]]))

            t0 = t.reshape(3, 2)
            t1 = t
            t2 = torch.randn(4, 5)
            nt = constructor([[t0, t1], [t2]])
            self.assertEqual(nt.unsqueeze(0), constructor([[[t0, t1], [t2]]]))
            self.assertEqual(nt.unsqueeze(
                1), constructor([[[t0, t1]], [[t2]]]))
            self.assertEqual(nt.unsqueeze(2), constructor(
                [[t0.reshape(1, 3, 2), t1.reshape(1, 2, 3)], [t2.reshape(1, 4, 5)]]))
            self.assertEqual(nt.unsqueeze(3), constructor(
                [[t0.reshape(3, 1, 2), t1.reshape(2, 1, 3)], [t2.reshape(4, 1, 5)]]))
            self.assertEqual(nt.unsqueeze(4), constructor(
                [[t0.reshape(3, 2, 1), t1.reshape(2, 3, 1)], [t2.reshape(4, 5, 1)]]))

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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
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

    @unittest.skip("Requires autograd support")
    def test_softmax_1(self):
        ts = [[], []]
        nt = ntnt(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Requires autograd support")
    def test_softmax_2(self):
        t0 = torch.randn(3)
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Requires autograd support")
    def test_softmax_3(self):
        t0 = torch.randn(3, 2, 1)
        t1 = torch.randn(2, 3, 1)
        t2 = torch.randn(3, 1, 2)
        ts = [[t0, t1], [t2]]
        nt = ntnt(ts)
        self._test_softmax(ts, nt)

    @unittest.skip("Requires autograd support")
    def test_softmax_4(self):
        ts = torch.randn(6, 4, 3, 2, 5)
        ts = list(map(lambda x: x.unbind(), ts.unbind()))
        nt = ntnt(ts)
        self._test_softmax(ts, nt)


if __name__ == "__main__":
    unittest.main()
