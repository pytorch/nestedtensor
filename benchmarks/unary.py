import torch
import nestedtensor
import utils

import random


# RAND_INTS = [random.randint(10, 30) for _ in range(2000)] # Performance tanks hard for lots of small Tensors as expected

def run_benchmarks(tensors):

    def gen_t_cos():
        tensor = torch.cat([tensor.reshape(-1) for tensor in tensors])
    
        def t():
            tensor.cos_()
        return t
    
    
    def gen_t_loop_cos():
    
        def t_loop():
            for t in tensors:
                t.cos_()
        return t_loop
    
    
    def gen_nt_cos():
        nested_tensor = nestedtensor.nested_tensor(tensors)
    
        def nt():
            nested_tensor.cos_()
        return nt
    
    
    def gen_ant_cos():
        nested_tensor = nestedtensor.as_nested_tensor(tensors)
    
        def ant():
            nested_tensor.cos_()
        return ant

    print(utils.benchmark_fn(gen_t_cos()))
    print(utils.benchmark_fn(gen_t_loop_cos()))
    print(utils.benchmark_fn(gen_nt_cos()))
    print(utils.benchmark_fn(gen_ant_cos()))


if __name__ == "__main__":
    RAND_INTS = [random.randint(100, 300) for _ in range(200)]
    print("medium cpu")
    run_benchmarks([torch.rand(i, 2560) for i in RAND_INTS])
    print("medium cuda")
    run_benchmarks([torch.rand(i, 2560).cuda() for i in RAND_INTS])
