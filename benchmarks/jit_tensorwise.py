import torch
import nestedtensor
import utils
import time


@nestedtensor._C.jit_tensorwise()
@torch.jit.script
def f(i, w):
    return torch.conv2d(i, w)

def loop_f(inp1, w):
    for inp in inp1:
        torch.conv2d(inp, w)


if __name__ == "__main__":
    w = torch.randn(64, 3, 9, 9).cuda()
    inp1 = list(torch.randn(128, 1, 3, 16, 16).cuda().unbind())
    inp3 = nestedtensor.as_nested_tensor(inp1)._impl
    # print(sum(inp.numel() for inp in inp1))
    # print(inp3.numel())

    fc = nestedtensor._C.jit_tensorwise()(torch.conv2d)

    t0 = time.time()
    count = 0
    while(time.time() - t0 < 5.0):
        r2 = fc(inp3, w)
        torch.cuda.synchronize()
        count += 1
    print("jit: " + str(count))

    t0 = time.time()
    count = 0
    while(time.time() - t0 < 5.0):
        loop_f(inp1, w)
        torch.cuda.synchronize()
        count += 1
    print("for loop: " + str(count))

    
    # print(r.nested_size())

    # na = nestedtensor._C.jit_tensorwise()(torch.mul)

    # print("111")
    # out = nestedtensor.as_nested_tensor([torch.randn(1, 2)])
    # print(na(
    #     nestedtensor.as_nested_tensor([torch.randn(1, 2)])._impl,
    #     4.0,
    #     ))
    # print("222")
    # print('out')
    # print(out)

    # nv = nestedtensor._C.jit_tensorwise()(torch.mv)
    # print(nv(
    #     nestedtensor._C._ListNestedTensor([torch.randn(1, 2)]),
    #     nestedtensor._C._ListNestedTensor([torch.randn(2)]),
    #     ))

    # print("333")
    # print(na(
    #     torch.randn(1, 2),
    #     torch.randn(1, 2),
    #     ))
    # print("444")
