import torch
import nestedtensor
import utils

import random
random.seed(1010)

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(1000, 3000) for _ in range(20)]

TENSORS0 = [torch.rand(9, 245, 2560, requires_grad=True).cuda() for i in RAND_INTS]
TENSORS1 = [torch.rand(9, 2560, 245, requires_grad=True).cuda() for i in RAND_INTS]

def gen_t_matmul():
    tensor0 = torch.stack(TENSORS0)
    tensor1 = torch.stack(TENSORS1)

    def t():
        tensor0.requires_grad_()
        tensor1.requires_grad_()
        torch.matmul(tensor0, tensor1).sum().backward()
        tensor0.detach_()
        tensor1.detach_()
    return t


def gen_t_loop_matmul():
    tensors = [torch.rand(i, 2560).cuda() for i in RAND_INTS]

    def t_loop():
        for (t0, t1) in zip(TENSORS0, TENSORS1):
            torch.matmul(t0, t1).sum().backward()
            t0.grad = None
            t1.grad = None
    return t_loop


def gen_nt_matmul():
    nt0 = nestedtensor.nested_tensor(TENSORS0, device=torch.device('cuda'), dtype=torch.float, requires_grad=True)
    nt1 = nestedtensor.nested_tensor(TENSORS1, device=torch.device('cuda'), dtype=torch.float, requires_grad=True)

    def nt():
        torch.matmul(nt0, nt1).sum().backward()
    return nt


if __name__ == "__main__":
    # print(utils.benchmark_fn(gen_t_matmul()))
    # print(utils.benchmark_fn(gen_t_loop_matmul()))
    print(utils.benchmark_fn(gen_nt_matmul()))
