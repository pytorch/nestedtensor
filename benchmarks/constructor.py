from nestedtensor import torch
import utils

import random


def gen_list_nested_tensor_construction():
    tensors = [torch.rand(random.randint(500, 1500), 25600) for _ in range(20)]
    def _algorithm():
        nt = torch._ListNestedTensor(tensors)
    return _algorithm

if __name__ == "__main__":
    alg = gen_list_nested_tensor_construction()
    # print(utils.benchmark_fn(alg, use_cprofile=True))
    print(utils.benchmark_fn(alg))
