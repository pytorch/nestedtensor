import torch
import nestedtensor
import utils

import random

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]


def gen_nt_unbind():
    nested_tensor = nestedtensor.nested_tensor(
        [torch.rand(i, 2560) for i in RAND_INTS])

    def nt():
        nested_tensor.unbind()
    return nt


def gen_ant_unbind():
    nested_tensor = nestedtensor.as_nested_tensor(
        [torch.rand(i, 2560) for i in RAND_INTS])

    def ant():
        nested_tensor.unbind()
    return ant


def gen_nt_unbind_2():
    nested_tensor = nestedtensor.nested_tensor(
        [[torch.rand(i, 25) for i in RAND_INTS] for j in range(100)])

    def nt_2():
        [t.unbind() for t in nested_tensor.unbind()]
    return nt_2


def gen_ant_unbind_2():
    nested_tensor = nestedtensor.as_nested_tensor(
        [[torch.rand(i, 25) for i in RAND_INTS] for j in range(100)])

    def ant_2():
        [t.unbind() for t in nested_tensor.unbind()]
    return ant_2


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_nt_unbind()))
    print(utils.benchmark_fn(gen_ant_unbind()))
    print(utils.benchmark_fn(gen_nt_unbind_2()))
    print(utils.benchmark_fn(gen_ant_unbind_2()))
