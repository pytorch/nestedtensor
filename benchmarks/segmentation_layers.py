import torch
import nestedtensor as nt
import utils
import torch.nn.functional as F
import sys 
import random
import argparse
from itertools import product

class SegLayersBenchMark(object):
    def __init__(self, args):
        self.args = args
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
        self.conv2d = torch.nn.Conv2d(args.C, 3, kernel_size= (1, 1), bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(args.C, 1e-05, 0.1)
        self.batch_norm.eval()
        self.inter_size = 10

    def run(self):
        for h_var, w_var, seed, layer in product(self.args.HV, self.args.WV, self.args.seed, self.args.layers):
            inputs, targets = self.get_input(h_var, w_var, seed)
            #                                       inputs                                                 squeeze      bench func         func
            settings_map = {"relu_pad" :           [inputs,                                                 False,     self.pad_func,     torch.nn.functional.relu],
                            "conv2d_pad" :         [inputs,                                                 False,     self.pad_func,     self.conv2d],
                            "batch_norm_pad" :     [inputs,                                                 False,     self.pad_func,     self.batch_norm],
                            "max_pool2d_pad" :     [inputs,                                                 False,     self.pad_func,     self.max_pool2d],
                            "cross_entropy_pad" :  [(inputs, targets),                                      False,     self.pad_func,     torch.nn.functional.cross_entropy],
                            "dropout_pad" :        [inputs,                                                 False,     self.pad_func,     torch.nn.functional.dropout],
                            "interpolate_pad" :    [(inputs, self.inter_size),                              False,     self.pad_func,     torch.nn.functional.interpolate],

                            "relu_iter" :          [inputs,                                                 False,     self.iter_func,    torch.nn.functional.relu],
                            "conv2d_iter" :        [inputs,                                                 True,      self.iter_func,    self.conv2d],
                            "batch_norm_iter" :    [inputs,                                                 True,      self.iter_func,    self.batch_norm],
                            "max_pool2d_iter" :    [inputs,                                                 True,      self.iter_func,    self.max_pool2d],
                            "cross_entropy_iter" : [(inputs, targets),                                      True,      self.iter_func,    torch.nn.functional.cross_entropy],
                            "dropout_iter" :       [inputs,                                                 True,      self.iter_func,    torch.nn.functional.dropout],
                            "interpolate_iter" :   [(inputs, self.inter_size),                              False,     self.iter_func,    torch.nn.functional.interpolate],

                            "relu_nt" :            [nt.nested_tensor(inputs),                               False,     self.nt_func,      torch.nn.functional.relu],
                            "conv2d_nt" :          [nt.nested_tensor(inputs),                               False,     self.nt_func,      self.conv2d],
                            "batch_norm_nt" :      [nt.nested_tensor(inputs),                               False,     self.nt_func,      self.batch_norm],
                            "max_pool2d_nt" :      [nt.nested_tensor(inputs),                               False,     self.nt_func,      self.max_pool2d],
                            "cross_entropy_nt" :   [(nt.nested_tensor(inputs), nt.nested_tensor(targets)),  False,     self.nt_func,      torch.nn.functional.cross_entropy],
                            "dropout_nt" :         [nt.nested_tensor(inputs),                               False,     self.nt_func,      torch.nn.functional.dropout],
                            "interpolate_nt" :     [(nt.nested_tensor(inputs), self.inter_size),            False,     self.nt_func,      torch.nn.functional.interpolate],
                            }
            func = settings_map[layer][2]
            result = utils.benchmark_fn(func(settings_map[layer]), warmup=self.args.warm)
            print(layer, ",", result['avg_us'], ",", result['std_us'], ",", result['runs'], ",", h_var, ",", w_var, ",", seed)

    def get_max_size(self, obj, res=None):
        if res is None:
            res = [1]

        if isinstance(obj, list) or isinstance(obj, tuple):
            for o in obj:
                res = self.get_max_size(o, res)

        if isinstance(obj, nt.nested.nested.NestedTensor):
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

    def get_input(self, h_var, w_var, seed):
        inputs = []
        targets = []

        torch.manual_seed(seed)
        for i in range(self.args.N):
            h_delta = random.gauss(self.args.H, h_var)
            w_delta = random.gauss(self.args.H, w_var)
            h = int(self.args.H + h_delta)
            w = int(self.args.W + w_delta)
            inputs.append(torch.randn(self.args.C, h, w))
            targets.append(torch.randint(1, (h, w), dtype=torch.int64))

        return inputs, targets

    def pad_func(self, args):
        func = args[3]

        if type(args[0]) is tuple:
            input_a = torch.stack(self.pad_tensors_to_max_shape(args[0][0]))
            input_b = args[0][1]

            if type(args[0][1]) is list:
                input_b = torch.stack(self.pad_tensors_to_max_shape(args[0][1]))

            def run():
                func(input_a, input_b)

        else:
            input_a = torch.stack(self.pad_tensors_to_max_shape(args[0]))

            def run():
                func(input_a)

        return run

    def iter_func(self, args):
        to_squeeze = args[1]
        func = args[3]

        if type(args[0]) is tuple:
            input_a = args[0][0]
            input_b = args[0][1]

            if type(args[0][1]) is int:
                if to_squeeze:
                    def run():
                        for a in input_a:
                            func(a.unsqueeze(0), input_b).squeeze(0)
                else:
                    def run():
                        for a in input_a:
                            func(a, input_b)
            else:
                if to_squeeze:
                    def run():
                        for a, b in zip(input_a, input_b):
                            func(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
                else:
                    def run():
                        for a, b in zip(input_a, input_b):
                            func(a, b)
        else:
            input_a = args[0]

            if to_squeeze:
                def run():
                    for a in input_a:
                        func(a.unsqueeze(0)).squeeze(0)
            else:
                def run():
                    for a in input_a:
                        func(a)

        return run

    def nt_func(self, args):
        func = args[3]

        if type(args[0]) is tuple:
            def run():
                func(args[0][0], args[0][1])
        else:
            def run():
                func(args[0])

        return run

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', dest='layers', type=str, nargs='+')
    parser.add_argument('-N', dest='N', type=int)
    parser.add_argument('-C', dest='C', type=int)
    parser.add_argument('-H', dest='H', type=int)
    parser.add_argument('-W', dest='W', type=int)
    parser.add_argument('-HV', dest='HV', type=int, nargs='+')
    parser.add_argument('-WV', dest='WV', type=int, nargs='+')
    parser.add_argument('-S', dest='seed', type=int, nargs='+')
    parser.add_argument('-WARM', dest='warm', type=float)
    args = parser.parse_args()

    print("called with: ", args)
    benchmark_obj = SegLayersBenchMark(args)
    benchmark_obj.run()

if __name__ == "__main__":
    main(sys.argv[1:])
