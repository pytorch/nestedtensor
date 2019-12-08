import torch

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


a = torch.randn(1, 2)
b = torch.randn(2, 1)
n = torch._ListNestedTensor([[a, b], [b]])
print(torch.nested_tensor.jit_apply_function(n, my_fun))
