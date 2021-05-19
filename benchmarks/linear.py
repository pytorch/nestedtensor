import torch
import nestedtensor
import utils

import random
random.seed(1010)

BDIM=10

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(100, 300) for _ in range(BDIM)]

OUTDIM=256
GOALDIM=512

TENSORS0 = [torch.rand(i, OUTDIM).cuda() for i in RAND_INTS]

def gen_t_linear():
    nt0 = nestedtensor.nested_tensor(TENSORS0, device=torch.device('cuda'), dtype=torch.float)
    data, _ = nt0.to_tensor_mask()
    lin = torch.nn.Linear(OUTDIM, GOALDIM).cuda()

    def t():
        lin(data)
    return t


@torch.inference_mode()
def gen_nt_linear():
    nt0 = nestedtensor.nested_tensor(TENSORS0, device=torch.device('cuda'), dtype=torch.float)
    lin = torch.nn.Linear(OUTDIM, GOALDIM).cuda()

    def nt():
        lin(nt0)
        # print("nt0.size()")
        # print(nt0.size())
        # import sys; sys.exit(1)
    return nt


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_t_linear()))
    print(utils.benchmark_fn(gen_nt_linear()))
