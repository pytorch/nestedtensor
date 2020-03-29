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

    def test_conv2d(self):
        tensor1 = torch.rand(3, 128, 128)
        tensor2 = torch.rand(3, 300, 400) 
        inputs = [tensor1, tensor2]

        weight = torch.rand(3, 3, 7, 7)
        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            nt_res = [t for t in torch.nn.functional.conv2d(nt, weight).unbind()]
            tensor_res = [torch.nn.functional.conv2d(t.unsqueeze(0), weight).squeeze(0) for t in inputs]
            self.assertEqual(nt_res, tensor_res)

    def test_batch_norm(self):
        inputs = [
            torch.tensor([[[-0.5000]], [[0.5000]]]),
            torch.tensor([
                [
                    [-1.0000, 1.0000], [-0.2500, -0.5000]
                ],
                [
                    [0.2500, 0.5000],   [1.5000, -1.5000]
                ]
            ])
        ]

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            tensor_res = []
            for i in range(2):
                batch_norm = torch.nn.BatchNorm2d(2, 1e-05, 0.1)
                batch_norm.eval()
                t_res = batch_norm(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))

            batch_norm = torch.nn.BatchNorm2d(2, 1e-05, 0.1)
            batch_norm.eval()
            nt_res = batch_norm(nt)

            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_max_pool2d(self):
        inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            maxPool2d = torch.nn.MaxPool2d(3)

            tensor_res = []
            for i in range(2):
                t_res = maxPool2d(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))

            nt_res = maxPool2d(nt)

            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_relu(self):
        inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]
        nt = nestedtensor.nested_tensor(inputs)

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.relu(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        nt_res = torch.nn.functional.relu(nt)

        self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_cross_entropy(self):
        inputs = [
            torch.randn(3, 300, 300),
            torch.randn(3, 400, 400)
        ]

        targets = [
            torch.randint(1, (300, 300), dtype=torch.int64),
            torch.randint(1, (400, 400), dtype=torch.int64)
        ]

        for input_nt, target_nt in [(nestedtensor.nested_tensor(inputs), nestedtensor.nested_tensor(targets)),
                                    (nestedtensor.as_nested_tensor(inputs), nestedtensor.as_nested_tensor(targets))]:

            nt_res = torch.nn.functional.cross_entropy(input_nt, target_nt)

            tensor_res = []
            for i in range(2):
                t_res = torch.nn.functional.cross_entropy(inputs[i].unsqueeze(0).contiguous(), targets[i].unsqueeze(0))
                tensor_res.append(t_res.squeeze(0))

            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_dropout(self):
        inputs = [
            torch.randn(3, 128, 128),
            torch.randn(3, 300, 400)
        ]

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            tensor_res = []
            for i in range(2):
                t_res = torch.nn.functional.dropout(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))

            nt_res = torch.nn.functional.dropout(nt)

            self.assertEqual(nestedtensor.nested_tensor(tensor_res).size(), nt_res.size())

    def test_interpolate(self):
        inputs = [
            torch.randn(3, 200, 300),
            torch.randn(3, 300, 400)
        ]

        for nt in [nestedtensor.nested_tensor(inputs), nestedtensor.as_nested_tensor(inputs)]:
            # No size
            tensor_res = []
            for i in range(2):
                t_res = torch.nn.functional.interpolate(inputs[i].unsqueeze(0).contiguous(), inputs[i].unsqueeze(0).shape[-2])
                tensor_res.append(t_res.squeeze(0))

            nt_res = torch.nn.functional.interpolate(nt)
            self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

            # tuple/int size
            for size  in [(200, 200), 100]:
                tensor_res = []
                for i in range(2):
                    t_res = torch.nn.functional.interpolate(inputs[i].unsqueeze(0).contiguous(), size)
                    tensor_res.append(t_res.squeeze(0))

                nt_res = torch.nn.functional.interpolate(nt, size)
                self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

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

            nt1 = constructor([[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
            nt2 = constructor([[torch.randn(1, 2, 3), torch.randn(2, 1, 3)], [torch.randn(3, 2, 1)]])
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
            self.assertEqual(nt.squeeze(1), constructor([t.reshape(1, 2, 1, 3)]))
            self.assertEqual(nt.squeeze(2), constructor([[t.reshape(2, 1, 3)]]))
            self.assertEqual(nt.squeeze(4), constructor([[t.reshape(1, 2, 3)]]))


if __name__ == "__main__":
    unittest.main()
