import torch
import nestedtensor
import utils
import torch.nn.functional as F
import sys 
import random

class SegLayersBenchMark(object):
    def __init__(self, args):
        assert len(args) == 6
        print("Called with args: ")
        
        self.N = int(args[0])
        print("N = ", self.N)

        self.C = int(args[1])
        print("C = ", self.C)

        self.H = int(args[2])
        print("H = ", self.H)

        self.W = int(args[3])
        print("W = ", self.W)
        
        self.delta = float(args[4])
        print("delta = ", self.W)

        self.warmup = float(args[5])
        print("warmup = ", self.W)
        
        self.conv2d = torch.nn.Conv2d(self.C, 3, kernel_size= (1, 1), bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(self.C, 1e-05, 0.1)
        self.batch_norm.eval()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1))

    def get_max_size(self, obj, res=None):
        if res is None:
            res = [1]

        if isinstance(obj, list) or isinstance(obj, tuple):
            for o in obj:
                res = self.get_max_size(o, res)

        if isinstance(obj, nestedtensor.nested.nested.NestedTensor):
            tres = self.get_max_size(obj.unbind())
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

    def pad_tensor_to_shape(self, t, goal_shape):
        padd = ()
        tup = tuple(t.size())
        assert(t.dim() == len(goal_shape))
        for i in range(len(tup)):
            padd = (0, goal_shape[i] - tup[i]) + padd
        new_tensor = F.pad(t, padd)
        new_tensor = new_tensor.reshape(goal_shape)
        return new_tensor

    def pad_tensors_to_max_shape(self, inputs):
        max_shape = self.get_max_size(inputs)
        padded_inputs = [self.pad_tensor_to_shape(t, max_shape) for t in inputs]
        return padded_inputs

    def get_input(self, return_targets=False, pad=False):
        inputs = []
        targets = []
        for i in range(self.N):
            h_change = random.randint(self.delta * 10 * (-self.H), self.delta * 10 * self.H) / 10
            w_change = random.randint(self.delta * 10 * (-self.W), self.delta * 10 * self.W) / 10
            inputs.append(torch.randn(self.C, self.H, self.W))
            targets.append(torch.randint(1, (self.H, self.W), dtype=torch.int64))

        if pad:
            inputs = self.pad_tensors_to_max_shape(inputs)

        if return_targets:
            return inputs, targets

        return inputs

    #
    # relu
    #
    def relu_tensor_iter(self):
        inputs = self.get_input()

        def _relu_tensor_iter():
            for t in inputs:
                torch.nn.functional.relu(t)
        return _relu_tensor_iter

    def relu_tensor_pad(self):
        inputs = self.get_input(pad=True)
        stack = torch.stack(inputs)

        def _relu_tensor_pad():
            torch.nn.functional.relu(stack)

        return _relu_tensor_pad

    def relu_nt(self):
        inputs = self.get_input()
        nt = nestedtensor.nested_tensor(inputs)
        def _relu_nt():
            torch.nn.functional.relu(nt)
        
        return _relu_nt

    #
    # conv2d
    #
    def conv2d_tensor_iter(self):
        inputs = self.get_input()

        def _conv2d_tensor_iter():
            for t in inputs:
                self.conv2d(t.unsqueeze(0)).squeeze(0)
        return _conv2d_tensor_iter

    def conv2d_tensor_pad(self):
        inputs = self.get_input(pad=True)
        stack = torch.stack(inputs)

        def _conv2d_tensor_pad():
            self.conv2d(stack)

        return _conv2d_tensor_pad

    def conv2d_nt(self):
        inputs = self.get_input()
        nt = nestedtensor.nested_tensor(inputs)
        def _conv2d_nt():
            self.conv2d(nt)

        return _conv2d_nt

    #
    # batch_norm
    #
    def batch_norm_tensor_iter(self):
        inputs = self.get_input()

        def _batch_norm_tensor_iter():
            for t in inputs:
                self.batch_norm(t.unsqueeze(0)).squeeze(0)
        return _batch_norm_tensor_iter

    def batch_norm_tensor_pad(self):
        inputs = self.get_input(pad=True)
        stack = torch.stack(inputs)

        def _batch_norm_tensor_pad():
            self.batch_norm(stack)

        return _batch_norm_tensor_pad

    def batch_norm_nt(self):
        inputs = self.get_input()
        nt = nestedtensor.nested_tensor(inputs)
        def _batch_norm_nt():
            self.batch_norm(nt)

        return _batch_norm_nt

    #
    # max_pool2d
    #
    def max_pool2d_tensor_iter(self):
        inputs = self.get_input()

        def _max_pool2d_tensor_iter():
            for t in inputs:
                self.max_pool2d(t.unsqueeze(0)).squeeze(0)
        return _max_pool2d_tensor_iter

    def max_pool2d_tensor_pad(self):
        inputs = self.get_input(pad=True)
        stack = torch.stack(inputs)

        def _max_pool2d_tensor_pad():
            self.max_pool2d(stack)

        return _max_pool2d_tensor_pad

    def max_pool2d_nt(self):
        inputs = self.get_input()
        nt = nestedtensor.nested_tensor(inputs)
        def _max_pool2d_nt():
            self.max_pool2d(nt)

        return _max_pool2d_nt

    #
    # cross_entropy
    #
    def cross_entropy_tensor_iter(self):
        inputs, targets = self.get_input(return_targets=True)

        def _cross_entropy_tensor_iter():
            for a, b in zip(inputs, targets):
                torch.nn.functional.cross_entropy(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
        return _cross_entropy_tensor_iter

    def cross_entropy_tensor_pad(self):
        inputs, targets = self.get_input(return_targets=True, pad=True)
        i_stack = torch.stack(inputs)
        t_stack = torch.stack(targets)

        def _cross_entropy_tensor_pad():
            torch.nn.functional.cross_entropy(i_stack, t_stack)

        return _cross_entropy_tensor_pad

    def cross_entropy_nt(self):
        inputs, targets = self.get_input(return_targets=True, pad=True)
        i_nt = nestedtensor.nested_tensor(inputs)
        t_nt = nestedtensor.nested_tensor(targets)
        def _cross_entropy_nt():
            torch.nn.functional.cross_entropy(i_nt, t_nt)

        return _cross_entropy_nt


    #
    # dropout
    #
    def dropout_tensor_iter(self):
        inputs = self.get_input()

        def _dropout_tensor_iter():
            for t in inputs:
                torch.nn.functional.dropout(t.unsqueeze(0)).squeeze(0)
        return _dropout_tensor_iter

    def dropout_tensor_pad(self):
        inputs = self.get_input(pad=True)
        stack = torch.stack(inputs)

        def _dropout_tensor_pad():
            torch.nn.functional.dropout(stack)

        return _dropout_tensor_pad

    def dropout_nt(self):
        inputs = self.get_input()
        nt = nestedtensor.nested_tensor(inputs)
        def _dropout_nt():
            torch.nn.functional.dropout(nt)

        return _dropout_nt
        inputs = get_input_big_diff()
        nt = nestedtensor.nested_tensor(inputs)

        def _dropout_nt_big_diff():
            dropout(nt)

        return _dropout_nt_big_diff

    #
    # interpolate
    #
    def interpolate_tensor_iter(self):
        inputs = self.get_input()

        def _interpolate_tensor_iter():
            for t in inputs:
                torch.nn.functional.interpolate(t,  t.unsqueeze(0).shape[-2])
        return _interpolate_tensor_iter

    def interpolate_tensor_pad(self):
        inputs = self.get_input(pad=True)
        stack = torch.stack(inputs)

        def _interpolate_tensor_pad():
            torch.nn.functional.interpolate(stack, inputs[0].unsqueeze(0).shape[-2])

        return _interpolate_tensor_pad

    def interpolate_nt(self):
        inputs = self.get_input()
        nt = nestedtensor.nested_tensor(inputs)
        def _interpolate_nt():
            torch.nn.functional.interpolate(nt)

        return _interpolate_nt



def main(args):
    b = SegLayersBenchMark(args)

    print(utils.benchmark_fn(b.relu_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.relu_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.relu_nt(), warmup=b.warmup))
    print(utils.benchmark_fn(b.conv2d_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.conv2d_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.conv2d_nt(), warmup=b.warmup))
    print(utils.benchmark_fn(b.batch_norm_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.batch_norm_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.batch_norm_nt(), warmup=b.warmup))
    print(utils.benchmark_fn(b.max_pool2d_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.max_pool2d_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.max_pool2d_nt(), warmup=b.warmup))
    print(utils.benchmark_fn(b.cross_entropy_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.cross_entropy_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.cross_entropy_nt(), warmup=b.warmup))
    print(utils.benchmark_fn(b.dropout_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.dropout_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.dropout_nt(), warmup=b.warmup))
    print(utils.benchmark_fn(b.interpolate_tensor_iter(), warmup=b.warmup))
    print(utils.benchmark_fn(b.interpolate_tensor_pad(), warmup=b.warmup))
    print(utils.benchmark_fn(b.interpolate_nt(), warmup=b.warmup))

if __name__ == "__main__":
    main(sys.argv[1:])
