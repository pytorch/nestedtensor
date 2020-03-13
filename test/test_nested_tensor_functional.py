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
import torchvision

class TestFunctional(TestCase):

    def test_conv2d(self):
        tensor1 = torch.rand(3, 128, 128)
        tensor2 = torch.rand(3, 300, 400) 
        list_of_tensors = [tensor1, tensor2]

        weight = torch.rand(3, 3, 7, 7)
        nt = nestedtensor.nested_tensor(list_of_tensors)
        nt_res = [t for t in torch.nn.functional.conv2d(nt, weight).unbind()]
        tensor_res = [torch.nn.functional.conv2d(t.unsqueeze(0), weight).squeeze(0) for t in list_of_tensors]
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

        nt = nestedtensor.nested_tensor(inputs)

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

        nt = nestedtensor.nested_tensor(inputs)
        maxPool2d = torch.nn.MaxPool2d(3)

        tensor_res = []
        for i in range(2):
            t_res = maxPool2d(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        nt_res = maxPool2d(nt)

        self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_max_relu(self):
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
            torch.randint(1, (1, 300, 300), dtype=torch.int64),
            torch.randint(1, (1, 400, 400), dtype=torch.int64)
        ]

        input_nt = nestedtensor.nested_tensor(inputs)
        target_nt = nestedtensor.nested_tensor(targets)
        nt_res = torch.nn.functional.cross_entropy(input_nt, target_nt)

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.cross_entropy(inputs[i].unsqueeze(0).contiguous(), targets[i])
            tensor_res.append(t_res.squeeze(0))

        

        self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_dropout(self):
        inputs = [
            torch.randn(3, 128, 128),
            torch.randn(3, 300, 400)
        ]

        nt = nestedtensor.nested_tensor(inputs)

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.dropout(inputs[i].unsqueeze(0).contiguous())
            tensor_res.append(t_res.squeeze(0))

        nt_res = torch.nn.functional.dropout(nt)

        self.assertEqual(nestedtensor.nested_tensor(tensor_res).size(), nt_res.size())

    def test_interpolate(self):
        inputs = [
            torch.randn(3, 128, 128),
            torch.randn(3, 300, 400)
        ]

        sizes = [128, 300]
        nt = nestedtensor.nested_tensor(inputs)

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.interpolate(inputs[i].unsqueeze(0).contiguous(), inputs[i].unsqueeze(0).shape[-2])
            tensor_res.append(t_res.squeeze(0))

        nt_res = torch.nn.functional.interpolate(nt)
        nt_res2 = torch.nn.functional.interpolate(nt, sizes)
        self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)
        self.assertEqual(nt_res, nt_res2)

if __name__ == "__main__":
    unittest.main()
