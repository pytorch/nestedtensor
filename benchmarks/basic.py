from nestedtensor import torch
import utils

import random


def gen_list_nested_tensor_construction():
    tensors = [torch.rand(random.randint(500, 1500), 25600) for _ in range(20)]
    def _algorithm():
        torch._ListNestedTensor(tensors)
    return _algorithm

def gen_list_nested_tensor_unbind():
    nested_tensor = torch._ListNestedTensor([torch.rand(random.randint(500, 1500), 25600) for _ in range(20)])
    def _algorithm():
        nested_tensor.unbind()
    return _algorithm

if __name__ == "__main__":
    # print(utils.benchmark_fn(alg, use_cprofile=True))
    # alg = gen_list_nested_tensor_construction()
    # print(utils.benchmark_fn(alg))
    alg = gen_list_nested_tensor_unbind()
    print(utils.benchmark_fn(alg))
