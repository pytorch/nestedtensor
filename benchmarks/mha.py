import torch
import nestedtensor
import utils
import torchvision

import random

# Performance tanks hard for lots of small Tensors as expected
random.seed(1010)
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(100, 300) for _ in range(20)]
DEVICE = torch.device('cuda')

# (26, None, 256) (26, None, 256) (26, None, 256) torch.Size([256, 256]) torch.Size([256])
MODEL0 = torch.nn.MultiheadAttention(256, 8, dropout=0.1).to(DEVICE)
MODEL1 = nestedtensor.nn.MultiheadAttention(256, 8, dropout=0.1).to(DEVICE)

def gen_t_loop_mha():
    tensors = [torch.rand(1, i, 256).to(DEVICE) for i in RAND_INTS]

    def t_loop():
        for t in tensors:
            MODEL0(t, t, t, need_weights=False) #[0].sum().backward()
    return t_loop


def gen_nt_mha():
    nt0 = nestedtensor.nested_tensor(
        [torch.rand(i, 256) for i in RAND_INTS], device=DEVICE, dtype=torch.float)

    def nt():
        MODEL1(nt0, nt0, nt0, need_weights=False) #[0].sum().backward()
    return nt


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_nt_mha()))
    # print(utils.benchmark_fn(gen_t_loop_mha()))
