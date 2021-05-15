import torch
import nestedtensor
import utils
import torchvision

import random

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(100, 300) for _ in range(20)]

backbone = torchvision.models.resnet.__dict__['resnet50'](
    pretrained=True,
    replace_stride_with_dilation=[False, True, True])

return_layers = {'layer4': 'out'}
MODEL = torchvision.models._utils.IntermediateLayerGetter(
    backbone, return_layers=return_layers).cuda()


def gen_t_loop_segmentation():
    tensors = [torch.rand(1, 3, i, 256).cuda() for i in RAND_INTS]

    def t_loop():
        for t in tensors:
            MODEL(t)['out'].sum().backward()
    return t_loop


def gen_nt_segmentation():
    nested_tensor = nestedtensor.nested_tensor(
        [torch.rand(3, i, 256) for i in RAND_INTS], device=torch.device('cuda'), dtype=torch.float)

    def nt():
        MODEL(nested_tensor)['out'].sum().backward()
    return nt


if __name__ == "__main__":
    # print(utils.benchmark_fn(gen_t_loop_segmentation(), 10.0))
    print(utils.benchmark_fn(gen_nt_segmentation(), 2.0))
