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

class TestIntegration(TestCase):
    def test_segmentation_pretrained_test_only(self):
        t1 = torch.randn(3, 2, 2, requires_grad=True)
        t2 = torch.randn(3, 2, 2, requires_grad=True)
        tr1 = torch.randn(2, 2, requires_grad=True)
        tr2 = torch.randn(2, 2, requires_grad=True)
       
       
        nt_t1 = t1.detach().requires_grad_()
        nt_t2 = t2.detach().requires_grad_()
        nt_tr1 = tr1.detach().requires_grad_()
        nt_tr2 = tr2.detach().requires_grad_()
        nt_t1.retain_grad()
        nt_t2.retain_grad()
        nt_tr1.retain_grad()
        nt_tr2.retain_grad()

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

        output1 = model(t_input)
        output1 = output1['out']
        
        confmat.update(t_target.flatten(), output1.argmax(1).flatten())
        confmat.reduce_from_all_processes()

        # nt run
        nt_input = nestedtensor.nested_tensor([nt_t1, nt_t2], requires_grad=True)
        nt_target = nestedtensor.nested_tensor([nt_tr1, nt_tr2], requires_grad=True)
        confmat2 = ConfusionMatrix(num_classes)

        output2 = model(nt_input)
        output2 = output2['out']

        for a, b in zip(nt_target, output2):
            confmat2.update(a.flatten(), b.argmax(0).flatten())
        
        confmat2.reduce_from_all_processes()
        self.assertEqual(confmat.mat, confmat2.mat)

        # grad test
        output1_sum = output1[0].sum()
        output2_sum = output2[0].sum()
        self.assertEqual(output1_sum, output2_sum)

        output1_sum.backward()
        output2_sum.backward()

        self.assertEqual(t1.grad, nt_t1.grad)
        self.assertEqual(t2.grad, nt_t2.grad)


if __name__ == "__main__":
    unittest.main()
