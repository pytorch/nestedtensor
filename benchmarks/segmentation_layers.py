import torch
import nestedtensor
import utils

#
# relu
#
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

#
# conv2d
#
def conv2d_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]
    
    a = torch.stack(inputs)
    conv2d = torch.nn.Conv2d(30, 33, kernel_size=(3, 5), bias=False)

    def _conv2d_tensor():
        conv2d(a)
    return _conv2d_tensor

def conv2d_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    conv2d = torch.nn.Conv2d(30, 33, kernel_size=(3, 5), bias=False)
    nt = nestedtensor.nested_tensor(inputs)
    def _conv2d_nt():
        conv2d(nt)
    
    return _conv2d_nt

#
# batch_norm
#
def batch_norm_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]
    
    a = torch.stack(inputs)
    batch_norm = torch.nn.BatchNorm2d(30, 1e-05, 0.1)
    batch_norm.eval()

    def _batch_norm_tensor():
        batch_norm(a)
    return _batch_norm_tensor

def batch_norm_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    nt = nestedtensor.nested_tensor(inputs)
    batch_norm = torch.nn.BatchNorm2d(30, 1e-05, 0.1)
    batch_norm.eval()

    def _batch_norm_nt():
        batch_norm(nt)
    
    return _batch_norm_nt

#
# max_pool2d
#
def max_pool2d_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    a = torch.stack(inputs)
    maxPool2d = torch.nn.MaxPool2d(30)

    def _max_pool2d_tensor():
        maxPool2d(a)
    return _max_pool2d_tensor

def max_pool2d_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    maxPool2d = torch.nn.MaxPool2d(30)
    nt = nestedtensor.nested_tensor(inputs)
    def _max_pool2d_nt():
        maxPool2d(nt)

    return _max_pool2d_nt

#
# cross_entropy
#
def cross_entropy_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    targets = [
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
        ]

    a = torch.stack(inputs)
    b = torch.stack(targets)
    

    def _cross_entropy_tensor():
        torch.nn.functional.cross_entropy(a, b)
    return _cross_entropy_tensor

def cross_entropy_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    targets = [
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
            torch.randint(1, (500, 600), dtype=torch.int64),
        ]

    a = nestedtensor.nested_tensor(inputs)
    b = nestedtensor.nested_tensor(targets)

    def _cross_entropy_nt():
        torch.nn.functional.cross_entropy(a, b)
    return _cross_entropy_nt

#
# dropout
#
def dropout_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]
    
    a = torch.stack(inputs)

    def _dropout_tensor():
        torch.nn.functional.dropout(a)
    return _dropout_tensor

def dropout_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    nt = nestedtensor.nested_tensor(inputs)
    def _dropout_nt():
        torch.nn.functional.dropout(nt)
    
    return _dropout_nt

#
# interpolate
#
def interpolate_tensor():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]
    
    a = torch.stack(inputs)

    def _interpolate_tensor():
        torch.nn.functional.interpolate(a)
    return _interpolate_tensor

def interpolate_nt():
    inputs = [
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
            torch.randn(30, 500, 600),
        ]

    nt = nestedtensor.nested_tensor(inputs)
    def _interpolate_nt():
        torch.nn.functional.interpolate(nt)
    
    return _interpolate_nt


if __name__ == "__main__":
    print(utils.benchmark_fn(relu_tensor(), warmup=2.0))
    print(utils.benchmark_fn(relu_nt(), warmup=2.0))
    print(utils.benchmark_fn(conv2d_tensor(), warmup=2.0))
    print(utils.benchmark_fn(conv2d_nt(), warmup=2.0))
    print(utils.benchmark_fn(batch_norm_tensor(), warmup=2.0))
    print(utils.benchmark_fn(batch_norm_nt(), warmup=2.0))
    print(utils.benchmark_fn(max_pool2d_tensor(), warmup=2.0))
    print(utils.benchmark_fn(max_pool2d_nt(), warmup=2.0))
    print(utils.benchmark_fn(cross_entropy_tensor(), warmup=2.0))
    print(utils.benchmark_fn(cross_entropy_nt(), warmup=2.0))
    print(utils.benchmark_fn(dropout_tensor(), warmup=2.0))
    print(utils.benchmark_fn(dropout_nt(), warmup=2.0))
    print(utils.benchmark_fn(interpolate_tensor(), warmup=2.0))
    print(utils.benchmark_fn(interpolate_nt(), warmup=2.0))

