import torch
import nestedtensor
import utils

import random


# RAND_INTS = [random.randint(100, 300) for _ in range(200)]
RAND_INTS = [random.randint(10, 30) for _ in range(2000)] # Performance tanks hard for lots of small Tensors as expected


def gen_t_cos():
    tensor = torch.cat([torch.rand(i, 2560).reshape(-1) for i in RAND_INTS])

    def t():
        tensor.cos_()
    return t


def gen_t_loop_cos():
    tensors = [torch.rand(i, 2560) for i in RAND_INTS]

    def t_loop():
        for t in tensors:
            t.cos_()
    return t_loop


def gen_nt_cos():
    nested_tensor = nestedtensor.nested_tensor(
        [torch.rand(i, 2560) for i in RAND_INTS])

    def nt():
        nested_tensor.cos_()
    return nt


def gen_ant_cos():
    nested_tensor = nestedtensor.as_nested_tensor(
        [torch.rand(i, 2560) for i in RAND_INTS])

    def ant():
        nested_tensor.cos_()
    return ant


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_t_cos()))
    print(utils.benchmark_fn(gen_t_loop_cos()))
    print(utils.benchmark_fn(gen_nt_cos()))
    print(utils.benchmark_fn(gen_ant_cos()))
