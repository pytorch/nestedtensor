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


def run_benchmark(bsz, mean_i, mean_j, var, writer):
    RAND_INTS = [(int(random.gauss(mean_j, var)), int(
        random.gauss(mean_i, var))) for _ in range(bsz)]
    src_ = nestedtensor.nested_tensor(
        [torch.randn(NDIM * i * j).float().reshape(NDIM, i, j) for (i, j) in RAND_INTS], device=DEVICE, dtype=torch.float)
    src = []
    for i, s in enumerate(src_):
        src.append(i*len(s) + s)

    detr_nt_src = DETRNestedTensor.from_tensor_list(src)
    sparsity = int(detr_nt_src.decompose()[1].float().mean().item() * 10) / 10

    def gen_t_loop_mha(src):
        detr_nt_src = DETRNestedTensor.from_tensor_list(src)
        src, mask = detr_nt_src.decompose()
        src = src.flatten(2).permute(2, 0, 1).contiguous()
        mask = mask.flatten(1).contiguous()

        def te():
            MODEL(src, src, src, key_padding_mask=mask,
                  need_weights=False)

        return te

    def gen_nt_mha(src):
        src = nestedtensor.nested_tensor([t.flatten(1).permute(
            1, 0) for t in src], device=DEVICE, dtype=torch.float)

        def nt():
            MODEL(src, src, src, need_weights=False)

        return nt

    result_t = {**utils.benchmark_fn(gen_t_loop_mha(src), 5.0, cuda=True), "bsz": bsz,
                "sparsity": sparsity, "var": var, "mean_i": mean_i, "mean_j": mean_j}
    result_t["numel"] = sum([x.numel() for x in src_])
    result_t["numel_div_avg_us"] = result_t["numel"] / result_t["avg_us"]
    result_t["avg_ns_div_numel"] = result_t["avg_us"] / \
        result_t["numel"] * 1000
    writer.writerow(result_t)
    result_nt = {**utils.benchmark_fn(gen_nt_mha(src), 5.0, cuda=True),
                 "bsz": bsz, "sparsity": 0.0, "var": var, "mean_i": mean_i, "mean_j": mean_j}
    result_nt["numel"] = sum([x.numel() for x in src_])
    result_nt["numel_div_avg_us"] = result_nt["numel"] / result_nt["avg_us"]
    result_nt["avg_ns_div_numel"] = result_nt["avg_us"] / \
        result_nt["numel"] * 1000
    writer.writerow(result_nt)


if __name__ == "__main__":
    random.seed(1011)
    torch.manual_seed(1011)
    writer = csv.DictWriter(sys.stdout, fieldnames=[
                            "name", "avg_us", "std_us", "runs", "bsz", "sparsity",
                            "var", "mean_i", "mean_j", "numel", "numel_div_avg_us",
                            "avg_ns_div_numel"])
    writer.writeheader()
    for var in [float(i) / 10 for i in range(0, 100, 50)]:
        for batch_size in [2, 8, 16]:
            run_benchmark(batch_size, 30, 30, var, writer)
