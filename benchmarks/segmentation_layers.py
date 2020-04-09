import torch
import nestedtensor
import utils
import torch.nn.functional as F

N = 6
C = 30
H = 500
W = 600

N1 = 3
C1 = 30
H1 = 100
W1 = 200

N2 = 3
C2 = 30
H2 = 400
W2 = 800

N3 = 3
C3 = 30
H3 = 1000
W3 = 2000

WARM = 2.0

def get_max_size(obj, res=None):
    if res is None:
        res = [1]

    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            res = get_max_size(o, res)

    if isinstance(obj, nestedtensor.nested.nested.NestedTensor):
        tres = get_max_size(obj.unbind())
        while len(tres) > len(res):
                res.append(0)

        res = [max(i, j) for (i, j) in zip(res, tres)]

    if isinstance(obj, torch.Tensor):
        # scalar
        if obj.dim() == 0 and obj.numel() == 1:
            res = [1]
        else:
            while len(obj.size()) > len(res):
                res.append(0)

            res = [max(i, j) for (i, j) in zip(res, obj.size())]

    return res

def pad_tensor_to_shape(t, goal_shape):
    padd = ()
    tup = tuple(t.size())
    assert(t.dim() == len(goal_shape))
    for i in range(len(tup)):
        padd = (0, goal_shape[i] - tup[i]) + padd
    new_tensor = F.pad(t, padd)
    new_tensor = new_tensor.reshape(goal_shape)
    return new_tensor

def pad_tensors_to_max_shape(inputs):
    max_shape = get_max_size(inputs)
    padded_inputs = [pad_tensor_to_shape(t, max_shape) for t in inputs]
    return padded_inputs

def get_input_small_diff(pad=False):
    inputs = [torch.randn(C1, H1, W1) for _ in range(N1)]
    inputs2 = [torch.randn(C2, H2, W2) for _ in range(N2)]
    inputs = inputs + inputs2

    if pad:
        return pad_tensors_to_max_shape(inputs)
    return inputs

def get_input_big_diff(pad=False):
    inputs = [torch.randn(C1, H1, W1) for _ in range(N1)]
    inputs2 = [torch.randn(C3, H3, W3) for _ in range(N3)]
    inputs = inputs + inputs2

    if pad: 
        return pad_tensors_to_max_shape(inputs)
    return inputs

def get_targets_small_diff(pad = False):
    targets = [torch.randint(1, (H1, W1), dtype=torch.int64) for _ in range(N1)]
    targets2 = [torch.randint(1, (H2, W2), dtype=torch.int64) for _ in range(N2)]
    trg = targets + targets2

    if pad: 
        return pad_tensors_to_max_shape(trg)

    return trg

def get_targets_big_diff(pad = False):
    targets = [torch.randint(1, (H1, W1), dtype=torch.int64) for _ in range(N1)]
    targets2 = [torch.randint(1, (H3, W3), dtype=torch.int64) for _ in range(N3)]
    trg = targets + targets2

    if pad: 
        return pad_tensors_to_max_shape(trg)

    return trg

#
# relu
#
def relu_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    a = torch.stack(inputs)

    def _relu_tensor():
        torch.nn.functional.relu(a)
    return _relu_tensor

def relu_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    nt = nestedtensor.nested_tensor(inputs)
    def _relu_nt():
        torch.nn.functional.relu(nt)
    
    return _relu_nt

def relu_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    stack = torch.stack(inputs)

    def _relu_tensor_small_diff_pad():
        torch.nn.functional.relu(stack)
    return _relu_tensor_small_diff_pad

def relu_tensor_small_diff_iter():
    inputs = get_input_small_diff()

    def _relu_tensor_small_diff_iter():
        for t in inputs:
            torch.nn.functional.relu(t)
    return _relu_tensor_small_diff_iter

def relu_nt_small_diff():
    inputs = get_input_small_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _relu_nt_small_diff():
        torch.nn.functional.relu(nt)

    return _relu_nt_small_diff

def relu_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    stack = torch.stack(inputs)

    def _relu_tensor_big_diff_pad():
        torch.nn.functional.relu(stack)
    return _relu_tensor_big_diff_pad

def relu_tensor_big_diff_iter():
    inputs = get_input_big_diff()

    def _relu_tensor_big_diff_iter():
        for t in inputs:
            torch.nn.functional.relu(t)
    return _relu_tensor_big_diff_iter

def relu_nt_big_diff():
    inputs = get_input_big_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _relu_nt_big_diff():
        torch.nn.functional.relu(nt)

    return _relu_nt_big_diff


#
# conv2d
#
conv2d = torch.nn.Conv2d(30, 33, kernel_size= (3, 5), bias=False)

def conv2d_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    a = torch.stack(inputs)

    def _conv2d_tensor_tensors_same_size():
        conv2d(a)
    return _conv2d_tensor_tensors_same_size

def conv2d_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    nt = nestedtensor.nested_tensor(inputs)
    def _conv2d_nt_tensors_same_size():
        conv2d(nt)

    return _conv2d_nt_tensors_same_size

def conv2d_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    stack = torch.stack(inputs)
    
    def _conv2d_tensor_small_diff_pad():
        conv2d(stack)
    return _conv2d_tensor_small_diff_pad

def conv2d_tensor_small_diff_iter():
    inputs = get_input_small_diff()
    
    def _conv2d_tensor_small_diff_iter():
        for t in inputs:
            conv2d(t.unsqueeze(0)).squeeze(0)
    return _conv2d_tensor_small_diff_iter

def conv2d_nt_small_diff():
    inputs = get_input_small_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _conv2d_nt_small_diff():
        conv2d(nt)

    return _conv2d_nt_small_diff

def conv2d_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    stack = torch.stack(inputs)

    def _conv2d_tensor_big_diff_pad():
        conv2d(stack)
    return _conv2d_tensor_big_diff_pad

def conv2d_tensor_big_diff_iter():
    inputs = get_input_big_diff()

    def _conv2d_tensor_big_diff_iter():
        for t in inputs:
            conv2d(t.unsqueeze(0)).squeeze(0)
    return _conv2d_tensor_big_diff_iter

def conv2d_nt_big_diff():
    inputs = get_input_big_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _conv2d_nt_big_diff():
        conv2d(nt)

    return _conv2d_nt_big_diff


#
# batch_norm
#
batch_norm = torch.nn.BatchNorm2d(30, 1e-05, 0.1)

def batch_norm_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    a = torch.stack(inputs)
    batch_norm.eval()

    def _batch_norm_tensor():
        batch_norm(a)
    return _batch_norm_tensor

def batch_norm_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    nt = nestedtensor.nested_tensor(inputs)
    batch_norm.eval()

    def _batch_norm_nt():
        batch_norm(nt)
    
    return _batch_norm_nt

def batch_norm_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    stack = torch.stack(inputs)
    batch_norm.eval()

    def _batch_norm_tensor_small_diff_pad():
        batch_norm(stack)
    return _batch_norm_tensor_small_diff_pad

def batch_norm_tensor_small_diff_iter():
    inputs = get_input_small_diff()
    batch_norm.eval()

    def _batch_norm_tensor_small_diff_iter():
        for t in inputs:
            batch_norm(t.unsqueeze(0)).squeeze(0)
    return _batch_norm_tensor_small_diff_iter

def batch_norm_nt_small_diff():
    inputs = get_input_small_diff()
    nt = nestedtensor.nested_tensor(inputs)
    batch_norm.eval()

    def _batch_norm_nt_small_diff():
        batch_norm(nt)

    return _batch_norm_nt_small_diff

def batch_norm_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    stack = torch.stack(inputs)
    batch_norm.eval()

    def _batch_norm_tensor_big_diff_pad():
        batch_norm(stack)
    return _batch_norm_tensor_big_diff_pad

def batch_norm_tensor_big_diff_iter():
    inputs = get_input_big_diff()
    batch_norm.eval()

    def _batch_norm_tensor_big_diff_iter():
        for t in inputs:
            batch_norm(t.unsqueeze(0)).squeeze(0)
    return _batch_norm_tensor_big_diff_iter

def batch_norm_nt_big_diff():
    inputs = get_input_big_diff()
    nt = nestedtensor.nested_tensor(inputs)
    batch_norm.eval()

    def _batch_norm_nt_big_diff():
        batch_norm(nt)

    return _batch_norm_nt_big_diff

#
# max_pool2d
#
max_pool2d = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1))

def max_pool2d_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    a = torch.stack(inputs)

    def _max_pool2d_tensor():
        max_pool2d(a)
    return _max_pool2d_tensor

def max_pool2d_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    nt = nestedtensor.nested_tensor(inputs)

    def _max_pool2d_nt():
        max_pool2d(nt)

    return _max_pool2d_nt

def max_pool2d_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    stack = torch.stack(inputs)
    
    def _max_pool2d_tensor_small_diff_pad():
        max_pool2d(stack)
    return _max_pool2d_tensor_small_diff_pad

def max_pool2d_tensor_small_diff_iter():
    inputs = get_input_small_diff()

    def _max_pool2d_tensor_small_diff_iter():
        for t in inputs:
            max_pool2d(t)
    return _max_pool2d_tensor_small_diff_iter

def max_pool2d_nt_small_diff():
    inputs = get_input_small_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _max_pool2d_nt_small_diff():
        max_pool2d(nt)

    return _max_pool2d_nt_small_diff

def max_pool2d_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    stack = torch.stack(inputs)

    def _max_pool2d_tensor_big_diff_pad():
        max_pool2d(stack)
    return _max_pool2d_tensor_big_diff_pad

def max_pool2d_tensor_big_diff_iter():
    inputs = get_input_big_diff()

    def _max_pool2d_tensor_big_diff_iter():
        for t in inputs:
            max_pool2d(t)
    return _max_pool2d_tensor_big_diff_iter

def max_pool2d_nt_big_diff():
    inputs = get_input_big_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _max_pool2d_nt_big_diff():
        max_pool2d(nt)

    return _max_pool2d_nt_big_diff

#
# cross_entropy
#
cross_entropy = torch.nn.functional.cross_entropy
def cross_entropy_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    targets = [torch.randint(1, (H, W), dtype=torch.int64) for _ in range(N)]
    
    a = torch.stack(inputs)
    b = torch.stack(targets)
    

    def _cross_entropy_tensor():
        cross_entropy(a, b)
    return _cross_entropy_tensor

def cross_entropy_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    targets = [torch.randint(1, (H, W), dtype=torch.int64) for _ in range(N)]

    a = nestedtensor.nested_tensor(inputs)
    b = nestedtensor.nested_tensor(targets)

    def _cross_entropy_nt():
        cross_entropy(a, b)
    return _cross_entropy_nt

def cross_entropy_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    targets = get_targets_small_diff(pad=True)

    a = torch.stack(inputs)
    b = torch.stack(targets)

    def _cross_entropy_tensor_small_diff_pad():
        cross_entropy(a, b)
    return _cross_entropy_tensor_small_diff_pad

def cross_entropy_tensor_small_diff_iter():
    inputs = get_input_small_diff()
    targets = get_targets_small_diff()

    def _cross_entropy_tensor_small_diff_iter():
        for a, b in zip(inputs, targets):
            cross_entropy(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return _cross_entropy_tensor_small_diff_iter

def cross_entropy_nt_small_diff():
    inputs = get_input_small_diff()
    targets = get_targets_small_diff()

    nt_input = nestedtensor.nested_tensor(inputs)
    nt_target = nestedtensor.nested_tensor(targets)

    def _cross_entropy_nt_small_diff():
        cross_entropy(nt_input, nt_target)

    return _cross_entropy_nt_small_diff

def cross_entropy_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    targets = get_targets_big_diff(pad=True)

    a = torch.stack(inputs)
    b = torch.stack(targets)

    def _cross_entropy_tensor_big_diff_pad():
        cross_entropy(a, b)
    return _cross_entropy_tensor_big_diff_pad

def cross_entropy_tensor_big_diff_iter():
    inputs = get_input_big_diff()
    targets = get_targets_big_diff()

    def _cross_entropy_tensor_big_diff_iter():
        for a, b in zip(inputs, targets):
            cross_entropy(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return _cross_entropy_tensor_big_diff_iter

def cross_entropy_nt_big_diff():
    inputs = get_input_big_diff()
    targets = get_targets_big_diff()
    
    nt_input = nestedtensor.nested_tensor(inputs)
    nt_target = nestedtensor.nested_tensor(targets)

    def _cross_entropy_nt_big_diff():
        cross_entropy(nt_input, nt_target)

    return _cross_entropy_nt_big_diff

#
# dropout
#
dropout = torch.nn.functional.dropout

def dropout_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    a = torch.stack(inputs)

    def _dropout_tensor():
        dropout(a)
    return _dropout_tensor

def dropout_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    nt = nestedtensor.nested_tensor(inputs)
    def _dropout_nt():
        dropout(nt)

    return _dropout_nt

def dropout_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    stack = torch.stack(inputs)
    
    def _dropout_tensor_small_diff_pad():
        dropout(stack)
    return _dropout_tensor_small_diff_pad

def dropout_tensor_small_diff_iter():
    inputs = get_input_small_diff()
    
    def _dropout_tensor_small_diff_iter():
        for t in inputs:
            dropout(t.unsqueeze(0)).squeeze(0)
    return _dropout_tensor_small_diff_iter

def dropout_nt_small_diff():
    inputs = get_input_small_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _dropout_nt_small_diff():
        dropout(nt)

    return _dropout_nt_small_diff

def dropout_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    stack = torch.stack(inputs)

    def _dropout_tensor_big_diff_pad():
        dropout(stack)
    return _dropout_tensor_big_diff_pad

def dropout_tensor_big_diff_iter():
    inputs = get_input_big_diff()

    def _dropout_tensor_big_diff_iter():
        for t in inputs:
            dropout(t.unsqueeze(0)).squeeze(0)
    return _dropout_tensor_big_diff_iter

def dropout_nt_big_diff():
    inputs = get_input_big_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _dropout_nt_big_diff():
        dropout(nt)

    return _dropout_nt_big_diff

#
# interpolate
#
interpolate = torch.nn.functional.interpolate
def interpolate_tensor_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    a = torch.stack(inputs)

    def _interpolate_tensor():
        interpolate(a, inputs[0].unsqueeze(0).shape[-2])
    return _interpolate_tensor

def interpolate_nt_tensors_same_size():
    inputs = [torch.randn(C, H, W) for _ in range(N)]
    nt = nestedtensor.nested_tensor(inputs)
    def _interpolate_nt():
        interpolate(nt)

    return _interpolate_nt

def interpolate_tensor_small_diff_pad():
    inputs = get_input_small_diff(pad=True)
    stack = torch.stack(inputs)
    
    def _interpolate_tensor_small_diff_pad():
        interpolate(stack, inputs[0].unsqueeze(0).shape[-2])
    return _interpolate_tensor_small_diff_pad

def interpolate_tensor_small_diff_iter():
    inputs = get_input_small_diff()
    
    def _interpolate_tensor_small_diff_iter():
        for t in inputs:
            interpolate(t, t.unsqueeze(0).shape[-2])
    return _interpolate_tensor_small_diff_iter

def interpolate_nt_small_diff():
    inputs = get_input_small_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _interpolate_nt_small_diff():
        interpolate(nt)

    return _interpolate_nt_small_diff

def interpolate_tensor_big_diff_pad():
    inputs = get_input_big_diff(pad=True)
    stack = torch.stack(inputs)

    def _interpolate_tensor_big_diff_pad():
        interpolate(stack, inputs[0].unsqueeze(0).shape[-2])
    return _interpolate_tensor_big_diff_pad

def interpolate_tensor_big_diff_iter():
    inputs = get_input_big_diff()

    def _interpolate_tensor_big_diff_iter():
        for t in inputs:
            interpolate(t, t.unsqueeze(0).shape[-2])
    return _interpolate_tensor_big_diff_iter

def interpolate_nt_big_diff():
    inputs = get_input_big_diff()
    nt = nestedtensor.nested_tensor(inputs)

    def _interpolate_nt_big_diff():
        interpolate(nt)

    return _interpolate_nt_big_diff

if __name__ == "__main__":
    # relu
    print(utils.benchmark_fn(relu_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(relu_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(relu_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(relu_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(relu_nt_small_diff(), warmup=WARM))
    print(utils.benchmark_fn(relu_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(relu_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(relu_nt_big_diff(), warmup=WARM))

    # conv2d
    print(utils.benchmark_fn(conv2d_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_nt_small_diff(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(conv2d_nt_big_diff(), warmup=WARM))

    # batch_norm
    print(utils.benchmark_fn(batch_norm_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(batch_norm_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(batch_norm_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(batch_norm_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(batch_norm_nt_small_diff(), warmup=WARM))  
    print(utils.benchmark_fn(batch_norm_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(batch_norm_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(batch_norm_nt_big_diff(), warmup=WARM))

    # max_pool2d
    print(utils.benchmark_fn(max_pool2d_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_nt_small_diff(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(max_pool2d_nt_big_diff(), warmup=WARM))

    # cross_entropy
    print(utils.benchmark_fn(cross_entropy_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_nt_small_diff(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(cross_entropy_nt_big_diff(), warmup=WARM))

    # dropout
    print(utils.benchmark_fn(dropout_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(dropout_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(dropout_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(dropout_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(dropout_nt_small_diff(), warmup=WARM))
    print(utils.benchmark_fn(dropout_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(dropout_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(dropout_nt_big_diff(), warmup=WARM))

    # interpolate
    print(utils.benchmark_fn(interpolate_tensor_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_nt_tensors_same_size(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_tensor_small_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_tensor_small_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_nt_small_diff(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_tensor_big_diff_pad(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_tensor_big_diff_iter(), warmup=WARM))
    print(utils.benchmark_fn(interpolate_nt_big_diff(), warmup=WARM))