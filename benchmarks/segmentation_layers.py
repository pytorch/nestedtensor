import torch
import nestedtensor
import utils

def relu_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]
    
    a = torch.stack(inputs)

    def _relu_tensor():
        torch.nn.functional.relu(a)
    return _relu_tensor

def relu_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    nt = nestedtensor.nested_tensor(inputs)
    def _relu_nt():
        torch.nn.functional.relu(nt)
    
    return _relu_nt

if __name__ == "__main__":
    print(utils.benchmark_fn(relu_tensor(), warmup=2.0))
    print(utils.benchmark_fn(relu_nt(), warmup=2.0))