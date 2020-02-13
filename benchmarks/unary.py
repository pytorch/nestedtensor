import torch
import nestedtensor
import utils

import random


def gen_cos():
    nested_tensor = nestedtensor.nested_tensor([torch.rand(random.randint(500, 1500), 2560) for _ in range(20)])
    def _algorithm():
        nested_tensor.cos_()
    return _algorithm

if __name__ == "__main__":
    print(utils.benchmark_fn(gen_cos(), use_cprofile=True))
