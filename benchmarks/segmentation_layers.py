import torch
import nestedtensor
import utils
import torch.nn.functional as F
import sys
import random
import argparse
import itertools
import re


Benchmarks = {}

def register_benchmark(fn):
    Benchmarks[fn.__name__] = fn

#
# relu
#
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
        self.conv2d(nt)

    return _conv2d

#
# batch_norm
#
@register_benchmark
def batch_norm_tensor_iter(self):
    def _batch_norm_tensor_iter():
        for t in self.inputs:
            self.batch_norm(t.unsqueeze(0)).squeeze(0)

    return _batch_norm_tensor_iter

@register_benchmark
def batch_norm_tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _batch_norm_tensor_pad():
        self.batch_norm(tensor)

    return _batch_norm_tensor_pad

@register_benchmark
def batch_norm_nt(self):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _batch_norm_nt():
        self.batch_norm(nt)

    return _batch_norm_nt

#
# max_pool2d
#
@register_benchmark
def max_pool2d_tensor_iter(self):
    def _max_pool2d_tensor_iter():
        for t in self.inputs:
            self.max_pool2d(t.unsqueeze(0)).squeeze(0)

    return _max_pool2d_tensor_iter

@register_benchmark
def max_pool2d_tensor_pad(self):
    tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

    def _max_pool2d_tensor_pad():
        self.max_pool2d(tensor)

    return _max_pool2d_tensor_pad

@register_benchmark
def max_pool2d_nt(self):
    nt = nestedtensor.nested_tensor(self.inputs)

    def _max_pool2d_nt():
        self.max_pool2d(nt)

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

    def _interpolate_nt():
        torch.nn.functional.interpolate(nt)

    return _interpolate_nt

class SegLayersBenchMark(object):
    def __init__(self, args):
        self.args = args
        self.layers = {}

    def get_benchmark(self, channels, name):
        if name.startswith("conv2d"):
            m = re.match(r"conv2d_([a-z]+)_(\d+)x(\d+)", name)
            if m is None:
                raise ValueError("Unsupported parameterization for conv2d layer {}".format(name))
            benchmark_kind = m.group(1)
            print(benchmark_kind)
            k0 = int(m.group(2))
            k1 = int(m.group(3))
            layer = self.layers.setdefault(
                name, torch.nn.Conv2d(channels, 3, kernel_size=(k0, k1), bias=False)
            )
            return getattr(Benchmarks, "conv2d_" + benchmark_kind)(self, layer)
        if name.startswith("batch_norm"):
            return self.layers.setdefault(
                name, torch.nn.BatchNorm2d(channels, 1e-05, 0.1).eval()
            )
            return getattr(Benchmarks, name)(self)
        if name.startswith("max_pool2d"):
            return self.layers.setdefault(
                name,
                torch.nn.MaxPool2d(
                    kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1)
                ),
            )
            return getattr(Benchmarks, name)(self)
        return getattr(Benchmarks, name)(self)

    def run(self):
        for n, c, h, w, h_var, w_var, seed in itertools.product(
            self.args.N,
            self.args.C,
            self.args.H,
            self.args.W,
            self.args.HV,
            self.args.WV,
            self.args.seed,
        ):

            # generate inputs before iterating layers to have the same imput per layer
            self.inputs, self.targets = self.get_input(n, c, h, w, h_var, w_var, seed)

            benchmarks = []
            for layer in self.args.layers:
                try:
                    benchmark = self.get_benchmark(c, layer)
                except AttributeError:
                    raise ValueError("Benchmark {} is not supported. Available benchmarks are\n{}.".format(layer,
                        "\n".join(Benchmarks.keys())))
                benchmarks.append(benchmark)

            for benchmark in benchmarks:
                result = utils.benchmark_fn(benchmark, warmup=self.args.warm)
                result["N"] = n
                result["C"] = c
                result["H"] = h
                result["W"] = w
                result["h_var"] = h_var
                result["w_var"] = w_var
                result["seed"] = seed
                result["avg_us"] = int(result["avg_us"])
                result["std_us"] = int(result["std_us"])

                print(
                    ",".join(
                        str((str(key), result[key])) for key in sorted(result.keys())
                    )
                )

    def get_input(self, n, c, h, w, h_var, w_var, seed):
        inputs = []
        targets = []

        torch.manual_seed(seed)
        for i in range(n):
            h_res = max(1, int(h + random.gauss(h, h_var)))
            w_res = max(1, int(w + random.gauss(w, w_var)))
            inputs.append(torch.randn(c, h_res, w_res))
            targets.append(torch.randint(1, (h_res, w_res), dtype=torch.int64))

        return inputs, targets



def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", dest="layers", type=str, nargs="+")
    parser.add_argument("-N", dest="N", type=int, nargs="+")
    parser.add_argument("-C", dest="C", type=int, nargs="+")
    parser.add_argument("-H", dest="H", type=int, nargs="+")
    parser.add_argument("-W", dest="W", type=int, nargs="+")
    parser.add_argument("-HV", dest="HV", type=int, nargs="+")
    parser.add_argument("-WV", dest="WV", type=int, nargs="+")
    parser.add_argument("-S", dest="seed", type=int, nargs="+")
    parser.add_argument("-WARM", dest="warm", type=float, default=2.0)
    parser.add_argument("-verbose", dest="verbose", type=int, default=0)
    args = parser.parse_args()

    if args.verbose > 0:
        print("called with: ", args)
    benchmark_obj = SegLayersBenchMark(args)
    benchmark_obj.run()


if __name__ == "__main__":
    main(sys.argv[1:])
