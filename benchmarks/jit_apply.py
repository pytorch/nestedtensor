from nestedtensor import torch
import utils


def vmap(fn):
    def decorator(arg):
        if torch.is_tensor(arg):
            return fn(arg)
        else:
            def asd(x):
                return fn(x)
            return arg.jit_apply(torch.jit.script(asd))
    return decorator


@torch.jit.script
def my_fun(x):
    x = x + 1
    y = x.abs()
    return y

# print(e)


def gen_current():
    n = torch.as_nested_tensor(
        [torch.randn(256, 128).to(device='cuda') for _ in range(128)])

    def _algorithm():
        n1 = n + 1
        n1.abs()

    return _algorithm


def gen_jit():

    n = torch._ListNestedTensor(
        [torch.randn(256, 128).to(device='cuda') for _ in range(128)])

    @torch.jit.script
    def my_fun(x):
        x = x + 1
        y = x.abs()
        return y

    def _algorithm():
        torch.jit_apply_function(n, my_fun)

    return _algorithm


if __name__ == "__main__":
    # print(utils.benchmark_fn(alg, use_cprofile=True))
    # alg = gen_list_nested_tensor_construction()
    # print(utils.benchmark_fn(alg))
    # alg1 = gen_current()
    # print(utils.benchmark_fn(alg1))
    alg2 = gen_jit()
    print(utils.benchmark_fn(alg2))
