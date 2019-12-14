import torch
import nestedtensor
import utils


@nestedtensor._C.jit_tensorwise()
@torch.jit.script
def f(i, w):
    return torch.conv2d(i, w)


if __name__ == "__main__":
    r = f(nestedtensor._C._ListNestedTensor([torch.randn(1, 3, 10, 20)]),
        nestedtensor._C._ListNestedTensor([torch.randn(5, 3, 3, 3)]))
    
    print(r.nested_size())
