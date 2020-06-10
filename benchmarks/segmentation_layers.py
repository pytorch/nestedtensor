import torch
import nestedtensor
import utils
import torch.nn.functional as F
import sys
import random
import argparse
import itertools
import re
import csv


Benchmarks = {}

def register_benchmark(fn):
    Benchmarks[fn.__name__] = fn

#
# relu
#
@register_benchmark
def relu__tensor_iter(self):
    def _relu_tensor_iter():
        for t in self.inputs:
            torch.nn.functional.relu_(t)

    return _relu_tensor_iter

@register_benchmark
def relu__tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _relu_tensor_pad():
        torch.nn.functional.relu_(tensor)

    return _relu_tensor_pad

@register_benchmark
def relu__nt(self):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _relu_nt():
        torch.nn.functional.relu_(nt)

    return _relu_nt

@register_benchmark
def relu_tensor_iter(self):
    def _relu_tensor_iter():
        for t in self.inputs:
            torch.nn.functional.relu(t)

    return _relu_tensor_iter

@register_benchmark
def relu_tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _relu_tensor_pad():
        torch.nn.functional.relu(tensor)

    return _relu_tensor_pad

@register_benchmark
def relu_nt(self):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _relu_nt():
        torch.nn.functional.relu(nt)

    return _relu_nt

#
# conv2d
#
@register_benchmark
def conv2d_iter(self, module):
    def _conv2d_tensor_iter():
        for t in self.inputs:
            module(t.unsqueeze(0)).squeeze(0)

    return _conv2d_tensor_iter

@register_benchmark
def conv2d_pad(self, module):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _conv2d_tensor():
        module(tensor)

    return _conv2d_tensor

@register_benchmark
def conv2d_nt(self, module):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _conv2d():
        module(nt)

    return _conv2d

#
# batch_norm
#
@register_benchmark
def batch_norm_tensor_iter(self, module):
    def _batch_norm_tensor_iter():
        for t in self.inputs:
            module(t.unsqueeze(0)).squeeze(0)

    return _batch_norm_tensor_iter

@register_benchmark
def batch_norm_tensor_pad(self, module):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _batch_norm_tensor_pad():
        module(tensor)

    return _batch_norm_tensor_pad

@register_benchmark
def batch_norm_nt(self, module):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _batch_norm_nt():
        module(nt)

    return _batch_norm_nt

#
# max_pool2d
#
@register_benchmark
def max_pool2d_tensor_iter(self, module):
    def _max_pool2d_tensor_iter():
        for t in self.inputs:
            module(t.unsqueeze(0)).squeeze(0)

    return _max_pool2d_tensor_iter

@register_benchmark
def max_pool2d_tensor_pad(self, module):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _max_pool2d_tensor_pad():
        module(tensor)

    return _max_pool2d_tensor_pad

@register_benchmark
def max_pool2d_nt(self, module):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _max_pool2d_nt():
        module(nt)

    return _max_pool2d_nt

#
# cross_entropy
#
@register_benchmark
def cross_entropy_tensor_iter(self):
    def _cross_entropy_tensor_iter():
        for a, b in zip(self.inputs, self.targets):
            torch.nn.functional.cross_entropy(
                a.unsqueeze(0), b.unsqueeze(0)
            ).squeeze(0)

    return _cross_entropy_tensor_iter

@register_benchmark
def cross_entropy_tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()
    targets = torch.stack(self.targets)

    def _cross_entropy_tensor_pad():
        torch.nn.functional.cross_entropy(tensor, targets)

    return _cross_entropy_tensor_pad

@register_benchmark
def cross_entropy_nt(self):
    nt_input = nestedtensor.nested_tensor(self.inputs)
    nt_targets = nestedtensor.nested_tensor(self.targets)

    def _cross_entropy_nt():
        torch.nn.functional.cross_entropy(nt_input, nt_targets)

    return _cross_entropy_nt

#
# dropout
#
@register_benchmark
def dropout_tensor_iter(self):
    def _dropout_tensor_iter():
        for t in self.inputs:
            torch.nn.functional.dropout(t.unsqueeze(0)).squeeze(0)

    return _dropout_tensor_iter

@register_benchmark
def dropout_tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _dropout_tensor_pad():
        torch.nn.functional.dropout(tensor)

    return _dropout_tensor_pad

@register_benchmark
def dropout_nt(self):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _dropout_nt():
        torch.nn.functional.dropout(nt)

    return _dropout_nt

#
# interpolate
#
@register_benchmark
def interpolate_tensor_iter(self):
    def _interpolate_tensor_iter():
        for t in self.inputs:
            torch.nn.functional.interpolate(t, t.unsqueeze(0).shape[-2])

    return _interpolate_tensor_iter

@register_benchmark
def interpolate_tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _interpolate_tensor_pad():
        torch.nn.functional.interpolate(tensor, tensor[0].unsqueeze(0).shape[-2])

    return _interpolate_tensor_pad

@register_benchmark
def interpolate_nt(self):
    nt = nestedtensor.nested_tensor(self.inputs)
    input_shape = [y[-2:] for y in nt.nested_size().unbind()]
    def _interpolate_nt():
        torch.nn.functional.interpolate(nt, input_shape)

    return _interpolate_nt

class SegLayersBenchMark(object):
    def __init__(self, args):
        self.args = args
        self.layers = {}

    def get_benchmark(self, channels, name, cuda):
        layer = None
        if name.startswith("conv2d"):
            m = re.match(r"conv2d_([a-z]+)_(\d+)x(\d+)", name)
            if m is None:
                raise ValueError("Unsupported parameterization for conv2d layer {}".format(name))
            benchmark_kind = m.group(1)
            k0 = int(m.group(2))
            k1 = int(m.group(3))
            # Parameters chosen based on dominant settings in
            # https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/segmentation.py#L19
            layer = self.layers.setdefault(
                (name, channels, cuda), torch.nn.Conv2d(channels, channels, kernel_size=(k0, k1), dilation=2, bias=False)
            )
            name = "conv2d_" + benchmark_kind
        if name.startswith("batch_norm"):
            layer = self.layers.setdefault(
                (name, cuda), torch.nn.BatchNorm2d(channels, 1e-05, 0.1).eval()
            )
        if name.startswith("max_pool2d"):
            layer = self.layers.setdefault(
                (name, cuda),
                torch.nn.MaxPool2d(
                    kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1)
                ),
            )
        try:
            if cuda and layer is not None:
                layer.cuda()
            return Benchmarks[name](self) if layer is None else Benchmarks[name](self, layer)
        except KeyError:
            raise ValueError("Benchmark {} is not supported. Available benchmarks are\n{}.".format(layer,
                "\n".join(sorted(Benchmarks.keys()))))

    def run(self):
        params = itertools.product(
            self.args.cuda,
            self.args.N,
            self.args.C,
            self.args.H,
            self.args.W,
            self.args.seed,
        )
        if self.args.V:
            var_params = [(v, v) for v in self.args.V]
        else:
            var_params = itertools.product(self.args.HV, self.args.WV)
        params = [[p + v for v in var_params] for p in params]
        params = sum(params, [])
            
        writer = None
        i = 0
        for cuda, n, c, h, w, seed, h_var, w_var in params:
            # generate inputs before iterating layers to have the same imput per layer
            self.inputs, self.targets = self.get_input(cuda, n, c, h, w, h_var, w_var, seed)

            benchmarks = [(layer, self.get_benchmark(c, layer, cuda)) for layer in self.args.layers]
            for layer, benchmark in benchmarks:
                result = utils.benchmark_fn(benchmark, run_time=self.args.run_time, warmup=self.args.warmup)
                result["#"] = str(i) + "/" + str(len(benchmarks) * len(params))
                result["N"] = n
                result["C"] = c
                result["H"] = h
                result["W"] = w
                result["h_var"] = h_var
                result["w_var"] = w_var
                result["seed"] = seed
                result["avg_us"] = int(result["avg_us"])
                result["std_us"] = int(result["std_us"])
                result["name"] = layer
                result["cuda"] = cuda
                result["numel"] = sum(x.numel() for x in self.inputs)
                if writer is None and self.args.csv_log:
                    writer = csv.DictWriter(open(self.args.csv_log, 'w'), fieldnames=result.keys())
                    writer.writeheader()
                if writer is not None:
                    writer.writerow(result)
                print(",".join(str((str(key), result[key])) for key in sorted(result.keys())))
                i += 1

    def get_input(self, cuda, n, c, h, w, h_var, w_var, seed):
        inputs = []
        targets = []

        torch.manual_seed(seed)
        if cuda:
            torch.cuda.init()
        for i in range(n):
            h_res = max(1, int(random.gauss(h, h_var)))
            w_res = max(1, int(random.gauss(w, w_var)))
            input_i = torch.randn(c, h_res, w_res)
            target_i = torch.randint(1, (h_res, w_res), dtype=torch.int64)
            inputs.append(input_i.cuda() if cuda else input_i)
            targets.append(target_i.cuda() if cuda else target_i)
        if cuda:
            # Synchronize copy operations so they don't influence the benchmark
            torch.cuda.synchronize()

        return inputs, targets



def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", dest="layers", type=str, nargs="+")
    parser.add_argument("-N", dest="N", type=int, nargs="+")
    parser.add_argument("-C", dest="C", type=int, nargs="+")
    parser.add_argument("-H", dest="H", type=int, nargs="+")
    parser.add_argument("-W", dest="W", type=int, nargs="+")
    parser.add_argument("-HV", dest="HV", type=float, nargs="+")
    parser.add_argument("-WV", dest="WV", type=float, nargs="+")
    parser.add_argument("-V", dest="V", type=float, nargs="+")
    parser.add_argument("-S", dest="seed", type=int, nargs="+")
    parser.add_argument("--warmup", dest="warmup", type=float, default=2.0)
    parser.add_argument("--run-time", dest="run_time", type=float, default=5.0)
    parser.add_argument("--verbose", dest="verbose", type=int, default=0)
    parser.add_argument("--csv-log", dest="csv_log", type=str)
    parser.add_argument("--cuda", dest="cuda", type=str, nargs="+", default=["False"])
    args = parser.parse_args()
    for v in args.cuda:
        if v not in ["False", "True"]:
            raise ValueError("Argument --cuda may only be passed a list of True or False. Got {} instead.".format(args.cuda))
    args.cuda = [True if c == "True" else False for c in args.cuda]

    if args.V is not None:
        if (args.HV is not None or args.WV is not None):
            raise ValueError("If specifying variance for both H and W, arguments HV and WV must not be set.")
        args.HV = args.V
        args.WV = args.V

    if args.verbose > 0:
        print("called with: ", args)
    benchmark_obj = SegLayersBenchMark(args)
    benchmark_obj.run()


if __name__ == "__main__":
    main(sys.argv[1:])
