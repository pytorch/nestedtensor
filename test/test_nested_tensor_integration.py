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


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)


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
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

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
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


class TestIntegration(TestCase):
    @unittest.skipIf(
        not utils.internet_on(), "Cannot reach internet to download reference model."
    )
    def test_segmentation_pretrained_test_only(self):
        torch.manual_seed(0)
        t1 = torch.randn(3, 2, 2, requires_grad=True)
        t2 = torch.randn(3, 2, 2, requires_grad=True)
        tr1 = torch.randn(2, 2, requires_grad=True)
        tr2 = torch.randn(2, 2, requires_grad=True)

        model_name = "fcn_resnet101"
        num_classes = 21
        aux_loss = "store_true"
        model0 = torchvision.models.segmentation.__dict__[model_name](
            num_classes=num_classes, aux_loss=aux_loss, pretrained=True
        )
        model0.eval()


        # tensor run
        t_input = torch.stack([t1]) #, t2])
        t_target = torch.stack([tr1]) #, tr2])
        confmat = ConfusionMatrix(num_classes)

        output1 = model0(t_input)
        output1 = output1["out"]
        output1_sum = output1.sum()
        output1_sum.backward()

        # confmat.update(t_target.flatten(), output1.argmax(1).flatten())
        # confmat.reduce_from_all_processes()

        # nt run

        model1 = torchvision.models.segmentation.__dict__[model_name](
            num_classes=num_classes, aux_loss=aux_loss, pretrained=True
        )
        model1.eval()
        nt_t1 = t1.clone().detach()
        nt_t2 = t2.clone().detach()
        nt_tr1 = tr1.clone().detach()
        nt_tr2 = tr2.clone().detach()

        nt_input = ntnt([nt_t1]) # , nt_t2])
        nt_target = ntnt([nt_tr1]) # , nt_tr2])
        confmat2 = ConfusionMatrix(num_classes)

        output2 = model1(nt_input)
        output2 = output2["out"]
        # print("nt_input.requires_grad")
        # print(nt_input.requires_grad)
        # print("output2.requires_grad")
        # print(output2.requires_grad)

        # for a, b in zip(nt_target, output2):
        #     confmat2.update(a.flatten(), b.argmax(0).flatten())

        # confmat2.reduce_from_all_processes()
        # self.assertEqual(confmat.mat, confmat2.mat)

        # grad test
        output2_sum = output2.sum()
        # print('output1_sum')
        # print(output1_sum)
        # print('output2_sum')
        # print(output2_sum)
        # print('output2_sum.requires_grad')
        # print(output2_sum.requires_grad)
        self.assertEqual(output1_sum, output2_sum)
        # print(model1)
        output2_sum.backward()

        a = list(model0.named_parameters())
        b = list(model1.named_parameters())
        for (n0, p0), (n1, p1) in zip(a, b):
            # print((n0, n1))
            if (p1.grad is None) and (p0.grad is not None):
                # print("IS NONE")
                continue
            if p0.grad is None:
                continue
            # print("p0gradsum: ", p0.grad.sum())
            # print("p1gradsum: ", p1.grad.sum())
            self.assertEqual(p0.grad, p1.grad)
        # print(list(filter(lambda x: x is not None, iter(n if p.grad is None else None for (n, p) in model0.named_parameters()))))
        # print(list(filter(lambda x: x is not None, iter(n if p.grad is None else None for (n, p) in model1.named_parameters()))))

        self.assertEqual(t1.grad, nt_input.grad[0])
        # self.assertEqual(t2.grad, nt_input.grad[1])


if __name__ == "__main__":
    unittest.main()
