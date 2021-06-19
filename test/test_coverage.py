import torch
import nestedtensor
import unittest
from torch.nn import functional as F
from torch import nn

from utils_test_case import TestCase


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)
def ntnt_nograd(x): return nestedtensor.nested_tensor(x, requires_grad=False)


# Various smoke tests to confirm coverage of an operator

class TestCoverage(TestCase):

    @unittest.skip("Fails for strange reason")
    @torch.inference_mode()
    def test_issues_313(self):
        # Based on https://github.com/pytorch/nestedtensor/issues/313

        def model(x):
            torch.manual_seed(20)
            linear = nn.Linear(9, 64)
            norm = nn.BatchNorm1d(64).eval()
            # 3 voxel with 40, 50 and 90 points respectively
            x = linear(x)
            x = norm(x.transpose(2, 1).contiguous()
                     ).transpose(2, 1).contiguous()
            x = F.relu(x)
            return torch.max(x, dim=1, keepdim=True)[0]

        inputs = [torch.randn(i, 9) for i in [40, 50, 90]]
        model(ntnt_nograd(inputs))

        inputs = [torch.randn(30, 9) for _ in range(3)]
        x0 = model(ntnt_nograd(inputs))
        x1 = model(torch.stack(inputs))
        self.assertEqual(torch.stack(x0.unbind()), x1)

    @unittest.skip("Fails for strange reason")
    @torch.inference_mode()
    def test_pytorch_commit_56017(self):
        # Based on https://github.com/pytorch/nestedtensor/issues/313

        nn.Linear(9, 64)
        # inputs = [torch.randn(i, 3) for i in [4, 5, 9]]
        # x0 = ntnt_nograd(inputs)
        # print(x0)
        # del inputs
        # x0 = x0 + x0
        # print(x0)


if __name__ == "__main__":
    unittest.main()
