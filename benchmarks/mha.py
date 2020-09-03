import torch
import sys
import csv
import nestedtensor
import utils
import torchvision
from torch.nn import functional as F

import random


class DETRNestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(
            *args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s)
                             for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1],
                        : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError('not supported')
        return cls(tensor, mask)


# Performance tanks hard for lots of small Tensors as expected
DEVICE = torch.device('cuda')
NDIM = 256
NHEAD = 8
MODEL = torch.nn.MultiheadAttention(NDIM, NHEAD).to(DEVICE).eval()


def run_benchmark(bsz, low, high, autograd, writer):
    RAND_INTS = [(random.randint(low, high), random.randint(low, high)) for _ in range(bsz)]
    src_ = nestedtensor.nested_tensor(
        [torch.arange(NDIM * i * j).float().reshape(NDIM, i, j) for (i, j) in RAND_INTS], device=DEVICE, dtype=torch.float)
    src = []
    for i, s in enumerate(src_):
        src.append(i*len(s) + s)

    detr_nt_src = DETRNestedTensor.from_tensor_list(src)
    sparsity = detr_nt_src.decompose()[1].float().mean().item()

    def gen_t_loop_mha(src):
        detr_nt_src = DETRNestedTensor.from_tensor_list(src)
        src, mask = detr_nt_src.decompose()
        src = src.flatten(2).permute(2, 0, 1).contiguous()
        mask = mask.flatten(1).contiguous()
        if autograd:
            src.requires_grad_()

        def te():
            if autograd:
                MODEL(src, src, src, key_padding_mask=mask,
                      need_weights=False)[0].sum().backward()
            MODEL(src, src, src, key_padding_mask=mask,
                      need_weights=False)

        return te

    def gen_nt_mha(src):
        src = nestedtensor.nested_tensor([t.flatten(1).permute(
            1, 0) for t in src], device=DEVICE, dtype=torch.float, requires_grad=True)

        def nt():
            if autograd:
                MODEL(src, src, src, need_weights=False)[0].sum().backward()
            MODEL(src, src, src, need_weights=False)

        return nt

    result_t = {**utils.benchmark_fn(gen_t_loop_mha(src), 0.1), "bsz": bsz, "sparsity": sparsity, "autograd": autograd}
    writer.writerow(result_t)
    result_nt = {**utils.benchmark_fn(gen_nt_mha(src), 0.1), "bsz": bsz, "sparsity": 0.0, "autograd": autograd}
    writer.writerow(result_nt)


if __name__ == "__main__":
    random.seed(1011)
    torch.manual_seed(1011)
    writer = csv.DictWriter(sys.stdout, fieldnames=["name", "avg_us", "std_us", "runs", "bsz", "sparsity", "autograd"])
    writer.writeheader()
    for autograd in [True, False]:
        for batch_size in [2, 4, 8]:
            for (i, j) in [(30, 30), (29, 31), (28, 32), (26, 34), (24, 36)]:
                run_benchmark(batch_size, i, j, autograd, writer)
