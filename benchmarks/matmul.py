import torch
import nestedtensor
import utils

import random
random.seed(1010)

BDIM=10

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]

OUTDIM=256

TENSORS0 = [torch.rand(i, OUTDIM).cuda() for i in RAND_INTS]

def gen_t_matmul():
    nt0 = nestedtensor.nested_tensor(TENSORS0, device=torch.device('cuda'), dtype=torch.float)
    data, _ = nt0.to_tensor_mask()
    t1 = torch.randn(OUTDIM, 512).cuda()

    def t():
        torch.matmul(data, t1)
    return t


@torch.inference_mode()
def gen_nt_matmul():
    nt0 = nestedtensor.nested_tensor(TENSORS0, device=torch.device('cuda'), dtype=torch.float)
    t1 = torch.randn(OUTDIM, 512).cuda()

    def nt():
        torch.matmul(nt0, t1)
    return nt


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_t_matmul()))
    print(utils.benchmark_fn(gen_nt_matmul()))
