import torch
import nestedtensor
import utils
import torch.nn.functional as F
import sys
import random
import argparse
import itertools


class SegLayersBenchMark(object):
    def __init__(self, args):
        self.args = args
        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1)
        )
        self.conv2d = torch.nn.Conv2d(args.C, 3, kernel_size=(1, 1), bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(args.C, 1e-05, 0.1)
        self.batch_norm.eval()

    def run(self):
        for h_var, w_var, seed, layer in itertools.product(
            self.args.HV, self.args.WV, self.args.seed, self.args.layers
        ):
            benchmark = getattr(self, layer)
            self.inputs, self.targets = self.get_input(h_var, w_var, seed)

            result = utils.benchmark_fn(benchmark(), warmup=self.args.warm)
            result["H"] = self.args.H
            result["W"] = self.args.W
            result["h_var"] = h_var
            result["w_var"] = w_var
            result["seed"] = seed
            result["avg_us"] = int(result["avg_us"])
            result["std_us"] = int(result["std_us"])
            print(",".join(str((str(n), result[n])) for n in sorted(result.keys())))

    def get_input(self, h_var, w_var, seed):
        inputs = []
        targets = []

        torch.manual_seed(seed)
        for i in range(self.args.N):
            h = max(1, int(self.args.H + random.gauss(self.args.H, h_var)))
            w = max(1, int(self.args.W + random.gauss(self.args.W, w_var)))
            inputs.append(torch.randn(self.args.C, h, w))
            targets.append(torch.randint(1, (h, w), dtype=torch.int64))

        return inputs, targets

    #
    # relu
    #
    def relu_tensor_iter(self):
        def _relu_tensor_iter():
            for t in self.inputs:
                torch.nn.functional.relu(t)

        return _relu_tensor_iter

    def relu_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

        def _relu_tensor_pad():
            torch.nn.functional.relu(tensor)

        return _relu_tensor_pad

    def relu_nt(self):
        nt = nestedtensor.nested_tensor(self.inputs)

        def _relu_nt():
            torch.nn.functional.relu(nt)

        return _relu_nt

    #
    # conv2d
    #
    def conv2d_tensor_iter(self):
        def _conv2d_tensor_iter():
            for t in self.inputs:
                self.conv2d(t.unsqueeze(0)).squeeze(0)

        return _conv2d_tensor_iter

    def conv2d_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

        def _conv2d_tensor_pad():
            self.conv2d(tensor)

        return _conv2d_tensor_pad

    def conv2d_nt(self):
        nt = nestedtensor.nested_tensor(self.inputs)

        def _conv2d_nt():
            self.conv2d(nt)

        return _conv2d_nt

    #
    # batch_norm
    #
    def batch_norm_tensor_iter(self):
        def _batch_norm_tensor_iter():
            for t in self.inputs:
                self.batch_norm(t.unsqueeze(0)).squeeze(0)

        return _batch_norm_tensor_iter

    def batch_norm_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

        def _batch_norm_tensor_pad():
            self.batch_norm(tensor)

        return _batch_norm_tensor_pad

    def batch_norm_nt(self):
        nt = nestedtensor.nested_tensor(self.inputs)

        def _batch_norm_nt():
            self.batch_norm(nt)

        return _batch_norm_nt

    #
    # max_pool2d
    #
    def max_pool2d_tensor_iter(self):
        def _max_pool2d_tensor_iter():
            for t in self.inputs:
                self.max_pool2d(t.unsqueeze(0)).squeeze(0)

        return _max_pool2d_tensor_iter

    def max_pool2d_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

        def _max_pool2d_tensor_pad():
            self.max_pool2d(tensor)

        return _max_pool2d_tensor_pad

    def max_pool2d_nt(self):
        nt = nestedtensor.nested_tensor(self.inputs)

        def _max_pool2d_nt():
            self.max_pool2d(nt)

        return _max_pool2d_nt

    #
    # cross_entropy
    #
    def cross_entropy_tensor_iter(self):
        def _cross_entropy_tensor_iter():
            for a, b in zip(self.inputs, self.targets):
                torch.nn.functional.cross_entropy(
                    a.unsqueeze(0), b.unsqueeze(0)
                ).squeeze(0)

        return _cross_entropy_tensor_iter

    def cross_entropy_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()
        targets = torch.stack(self.targets)

        def _cross_entropy_tensor_pad():
            torch.nn.functional.cross_entropy(tensor, targets)

        return _cross_entropy_tensor_pad

    def cross_entropy_nt(self):
        nt_input = nestedtensor.nested_tensor(self.inputs)
        nt_targets = nestedtensor.nested_tensor(self.targets)

        def _cross_entropy_nt():
            torch.nn.functional.cross_entropy(nt_input, nt_targets)

        return _cross_entropy_nt

    #
    # dropout
    #
    def dropout_tensor_iter(self):
        def _dropout_tensor_iter():
            for t in self.inputs:
                torch.nn.functional.dropout(t.unsqueeze(0)).squeeze(0)

        return _dropout_tensor_iter

    def dropout_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

        def _dropout_tensor_pad():
            torch.nn.functional.dropout(tensor)

        return _dropout_tensor_pad

    def dropout_nt(self):
        nt = nestedtensor.nested_tensor(self.inputs)

        def _dropout_nt():
            torch.nn.functional.dropout(nt)

        return _dropout_nt

    #
    # interpolate
    #
    def interpolate_tensor_iter(self):
        def _interpolate_tensor_iter():
            for t in self.inputs:
                torch.nn.functional.interpolate(t, t.unsqueeze(0).shape[-2])

        return _interpolate_tensor_iter

    def interpolate_tensor_pad(self):
        tensor, _ = nestedtensor.nested_tensor(self.inputs).to_tensor_mask()

        def _interpolate_tensor_pad():
            torch.nn.functional.interpolate(tensor, tensor[0].unsqueeze(0).shape[-2])

        return _interpolate_tensor_pad

    def interpolate_nt(self):
        nt = nestedtensor.nested_tensor(self.inputs)

        def _interpolate_nt():
            torch.nn.functional.interpolate(nt)

        return _interpolate_nt


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", dest="layers", type=str, nargs="+")
    parser.add_argument("-N", dest="N", type=int)
    parser.add_argument("-C", dest="C", type=int)
    parser.add_argument("-H", dest="H", type=int)
    parser.add_argument("-W", dest="W", type=int)
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
