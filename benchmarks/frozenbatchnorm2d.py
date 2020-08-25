import torch
import nestedtensor
import utils
import torchvision

import random

random.seed(1010)
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(100, 300) for _ in range(20)]

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        # print(scale.size())
        # print(bias.size())
        # print(type(scale))
        # print(type(bias))
        # print(x.nested_size())
        return (x * scale + bias).squeeze(1)

MODEL = FrozenBatchNorm2d(64).cuda()

def gen_t_loop_frozenbatchnorm2d():
    tensors = [torch.rand(64, i, 256).cuda() for i in RAND_INTS]

    def t_loop():
        for t in tensors:
            MODEL(t.unsqueeze(0))
    return t_loop


def gen_nt_frozenbatchnorm2d():
    nt0 = nestedtensor.nested_tensor(
        [torch.rand(64, i, 256).cuda() for i in RAND_INTS])

    def nt():
        MODEL(nt0)
    return nt


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_nt_frozenbatchnorm2d()))
    print(utils.benchmark_fn(gen_t_loop_frozenbatchnorm2d()))
