import torch
import nestedtensor
import utils
import torchvision
from torch.nn import functional as F

import random

# Performance tanks hard for lots of small Tensors as expected
DEVICE = torch.device('cpu')

# MODEL = torch.nn.MultiheadAttention(256, 8, dropout=0.1).to(DEVICE)
NDIM=4
BSZ=2
MODEL = torch.nn.MultiheadAttention(NDIM, 2, dropout=0.5).to(DEVICE).eval()
print(MODEL.in_proj_weight.data.fill_(1))
print(MODEL.in_proj_bias.data.fill_(1))
print(MODEL.out_proj.weight.data.fill_(1))
print(MODEL.out_proj.bias.data.fill_(1))


class DETRNestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError('not supported')
        # for name, x in xs.items():
        # print(tensor.shape[-2:])
        # tensor = tensor.
        # mask = F.interpolate(mask[None].float(), size=tensor.shape[-2:]).bool()[0]
        # print(mask.size())
        #    out[name] = NestedTensor(x, mask)
        return cls(tensor, mask)

    def __repr__(self):
        return repr(self.tensors)

def run_benchmark(shapes):
    # query = nestedtensor.nested_tensor(
    #     [torch.rand(100, 256) for i in RAND_INTS], device=DEVICE, dtype=torch.float)
    # key = nestedtensor.nested_tensor(
    #     [torch.rand(i, 256) for i in RAND_INTS], device=DEVICE, dtype=torch.float)
    # value = nestedtensor.nested_tensor(
    #     [torch.rand(i, 256) for i in RAND_INTS], device=DEVICE, dtype=torch.float)
    src_ = nestedtensor.nested_tensor(
        [torch.arange(NDIM * i * i).float().reshape(NDIM, i, i) for i in RAND_INTS], device=DEVICE, dtype=torch.float)
    src = []
    for i, s in enumerate(src_):
        src.append(i*len(s) + s)

    detr_nt_src = DETRNestedTensor.from_tensor_list(src)

    def gen_t_loop_mha(detr_nt_src):
        src, mask = detr_nt_src.decompose()
        print("____")
        print(src.size())
        print(mask.size())
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        print(src.size())
        print(mask.size())
        print(src)
        print(mask)
        print("____")
        # query_, query_mask = nestedtensor.nested_tensor(query_list).to_tensor_mask()
        # key_, key_mask = nestedtensor.nested_tensor(key_list).to_tensor_mask()
        # value_, value_mask = nestedtensor.nested_tensor(value_list).to_tensor_mask()

        # key_mask_bool = ~key_mask.bool()
        # value_mask_bool = ~value_mask.bool()

        # query = query_.transpose(0, 1)
        # key = key_.transpose(0, 1)
        # value = value_.transpose(0, 1)
    
        # result = MODEL(query, key, value, key_padding_mask=key_mask_bool, need_weights=False) #[0].sum().backward()
        result = MODEL(src, src, src, key_padding_mask=mask, need_weights=False) #[0].sum().backward()
        print(result[0])
        return None

        # def t_loop():
        #     for t in tensors:
        #         MODEL(t, t, t, need_weights=False) #[0].sum().backward()
        # return t_loop
    
    
    def gen_nt_mha(src):
        src = nestedtensor.nested_tensor([t.flatten(1).permute(1, 0) for t in src], device=DEVICE, dtype=torch.float)
        print(src.nested_size())
        print(src)
        print(MODEL(src, src, src, need_weights=False)[0])
        return None
    
        # def nt():
        #     MODEL(query, key, value, need_weights=False) #[0].sum().backward()
        # return nt

    print((gen_t_loop_mha(detr_nt_src)))
    print((gen_nt_mha(src)))

    # print(utils.benchmark_fn(gen_nt_mha(query, key, value)))
    # print(utils.benchmark_fn(gen_t_loop_mha(query, key, value)))

if __name__ == "__main__":
    random.seed(1011)
    torch.manual_seed(1011)
    # RAND_INTS = [random.randint(800, 1000) for _ in range(20)]
    # RAND_INTS = [random.randint(1, 2) for _ in range(BSZ)]
    RAND_INTS = [1, 2]
    run_benchmark(RAND_INTS)
