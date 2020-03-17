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

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

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
            torch.randn(3, 200, 300),
            torch.randn(3, 300, 400)
        ]

        nt = nestedtensor.nested_tensor(inputs)

        tensor_res = []
        for i in range(2):
            t_res = torch.nn.functional.interpolate(inputs[i].unsqueeze(0).contiguous(), inputs[i].unsqueeze(0).shape[-2])
            tensor_res.append(t_res.squeeze(0))

        nt_res = torch.nn.functional.interpolate(nt)
        self.assertEqual(nestedtensor.nested_tensor(tensor_res), nt_res)

    def test_segmentation_pretrained_test_only(self):
        t1 = torch.randn(3, 5, 7)
        t2 = torch.randn(3, 5, 7)
        tr1 = torch.randn(5, 7)
        tr2 = torch.randn(5, 7)

        model_name = 'fcn_resnet101'
        num_classes = 21
        aux_loss = 'store_true'
        model = torchvision.models.segmentation.__dict__[model_name](num_classes=num_classes,
                                                                     aux_loss=aux_loss,
                                                                     pretrained=True)
        model.eval()

        # tensor run
        t_input = torch.stack([t1, t2])
        t_target = torch.stack([tr1, tr2])
        confmat = ConfusionMatrix(num_classes)

        output = model(t_input)
        output = output['out']

        confmat.update(t_target.flatten(), output.argmax(1).flatten())
        confmat.reduce_from_all_processes()

        # nt run
        nt_input = nestedtensor.nested_tensor([t1, t2])
        nt_target = nestedtensor.nested_tensor([tr1, tr2])
        confmat2 = ConfusionMatrix(num_classes)

        output = model(nt_input)
        output = output['out']

        for a, b in zip(nt_target, output):
            confmat2.update(a.flatten(), b.argmax(0).flatten())
        
        confmat2.reduce_from_all_processes()
        self.assertEqual(confmat.mat, confmat2.mat)


if __name__ == "__main__":
    unittest.main()
