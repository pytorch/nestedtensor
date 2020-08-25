import torch
import nestedtensor
import utils
import torchvision

import random

random.seed(1010)
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(100, 300) for _ in range(20)]

# (26, None, 256) (26, None, 256) (26, None, 256) torch.Size([256, 256]) torch.Size([256])
MODEL0 = torch.nn.MultiheadAttention(256, 8, dropout=0.1).cuda()
MODEL1 = nestedtensor.nn.MultiheadAttention(256, 8, dropout=0.1).cuda()

def gen_t_loop_mha():
    tensors = [torch.rand(1, i, 256).cuda() for i in RAND_INTS]

    def t_loop():
        for t in tensors:
            MODEL0(t, t, t, need_weights=False)
    return t_loop


def gen_nt_mha():
    nt0 = nestedtensor.nested_tensor(
        [torch.rand(i, 256).cuda() for i in RAND_INTS])

    def nt():
        MODEL1(nt0, nt0, nt0, need_weights=False)
    return nt


if __name__ == "__main__":
<<<<<<< HEAD
    print(utils.benchmark_fn(gen_nt_mha()))
    print(utils.benchmark_fn(gen_t_loop_mha()))
=======
    print(utils.benchmark_fn(gen_nt_segmentation()))
    print(utils.benchmark_fn(gen_t_loop_segmentation()))
>>>>>>> master
