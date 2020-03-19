import torch
import nestedtensor
import nestedtensor
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
    n = nestedtensor.as_nested_tensor(
        [torch.randn(256, 128).to(device='cuda') for _ in range(128)])

    def _algorithm():
        n1 = n + 1
        n1.abs()

    return _algorithm


def gen_jit():

    n = nestedtensor._C._ListNestedTensor(
        [torch.randn(256, 128).to(device='cuda') for _ in range(128)])

    def gen_my_fun(scalar, tensor):
        @torch.jit.ignore
        def get_scalar():
            return scalar

        @torch.jit.ignore
        def get_tensor():
            return tensor

        @torch.jit.script
        def my_fun(x, y):
            x = x + get_scalar()
            x = x + get_tensor()
            y = y + x.abs()
            return y
        return my_fun
    my_fun = gen_my_fun(3.0, torch.randn(1).to(device='cuda'))

    def _algorithm_jit():
        nestedtensor._C.jit_apply_function((n, n), my_fun)

    return _algorithm_jit


if __name__ == "__main__":
    # print(utils.benchmark_fn(alg, use_cprofile=True))
    # alg = gen_list_nested_tensor_construction()
    # print(utils.benchmark_fn(alg))
    alg1 = gen_current()
    print(utils.benchmark_fn(alg1))
    alg2 = gen_jit()
    print(utils.benchmark_fn(alg2))
