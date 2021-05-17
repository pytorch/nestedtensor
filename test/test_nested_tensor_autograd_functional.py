import torch
import nestedtensor
import unittest
from utils_test_case import TestCase
import random
from frozen_batch_norm_2d import NTFrozenBatchNorm2d
from position_encoding import PositionEmbeddingSine
from joiner import Joiner


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)
def ntnt_nograd(x): return nestedtensor.nested_tensor(x)


class TestAutogradFunctional(TestCase):
    @unittest.skip("Requires autograd support")
    def test_nn_conv2d(self):
        def _test(Conv2d):
            inputs = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]

            # most of optional params
            conv2d = Conv2d()
            tensor_res = []
            for i in range(2):
                t_res = conv2d(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))
                t_res.sum().backward()
            layer_grad0 = [p.grad for (n, p) in conv2d.named_parameters()]

            conv2d.zero_grad()

            nt = ntnt(inputs)
            nt_res = conv2d(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in conv2d.named_parameters()]

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(nt.grad[0], inputs[0].grad)
            self.assertEqual(nt.grad[1], inputs[1].grad)

        _test(lambda: torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1), padding=(
            4, 2), padding_mode='zeros', dilation=1, groups=1, bias=True))
        _test(lambda: torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1), padding=(
            4, 2), padding_mode='zeros', dilation=1, groups=1, bias=False))
        _test(lambda: torch.nn.Conv2d(3, 33, kernel_size=3, stride=(2, 1)))
        _test(lambda: torch.nn.Conv2d(
            3, 33, kernel_size=(1, 1), stride=(1, 1), bias=False))

    @unittest.skip("Requires autograd support")
    def test_nn_linear(self):
        def _test(linear):
            inputs = [
                torch.randn(3, 10, requires_grad=True),
                torch.randn(3, 10, requires_grad=True)
            ]

            # most of optional params
            linear = linear()
            tensor_res = []
            for i in range(2):
                t_res = linear(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))
                t_res.sum().backward()
            layer_grad0 = [p.grad for (n, p) in linear.named_parameters()]

            linear.zero_grad()

            nt = ntnt(inputs)
            nt_res = linear(nt)
            nt_res.sum().backward()
            layer_grad1 = [p.grad for (n, p) in linear.named_parameters()]

            self.assertEqual(ntnt(tensor_res), nt_res)
            map(self.assertEqual, zip(layer_grad0, layer_grad1))
            self.assertEqual(nt.grad[0], inputs[0].grad)
            self.assertEqual(nt.grad[1], inputs[1].grad)

        _test(lambda: torch.nn.Linear(10, 6))

    @unittest.skip("Requires autograd support")
    def test_nn_batch_norm(self):
        def _test(BatchNorm2d, has_grad=True):
            inputs = torch.randn(5, 3, 18, 18, requires_grad=True)

            batch_norm = BatchNorm2d()

            t_res = batch_norm(inputs)
            t_res.sum().backward()
            layer_grad0 = [p.grad for (n, p) in batch_norm.named_parameters()]

            batch_norm.zero_grad()
            nt = ntnt(inputs.unbind())
            nt_res = batch_norm(nt)

            self.assertEqual(ntnt(t_res.unbind()), nt_res)
            if has_grad:
                nt_res.sum().backward()
                layer_grad1 = [p.grad for (
                    n, p) in batch_norm.named_parameters()]
                map(self.assertEqual, zip(layer_grad0, layer_grad1))
                self.assertEqual(nt.grad[0], inputs.grad[0])
                self.assertEqual(nt.grad[1], inputs.grad[1])
            else:
                self.assertRaisesRegex(
                    RuntimeError, "var.dim gradient not implemented yet.", lambda: nt_res.sum().backward())

        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=True), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=True).eval())
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=True, track_running_stats=False), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=False).eval(), False)

        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=False, track_running_stats=False), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=False, track_running_stats=False).eval(), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05,
                                           momentum=0.1, affine=False, track_running_stats=True), False)
        _test(lambda: torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1,
                                           affine=False, track_running_stats=True).eval())
        _test(lambda: torch.nn.BatchNorm2d(3), False)

    @unittest.skip("Requires autograd support")
    def test_nn_relu(self):
        inputs = [
            torch.randn(3, 500, 600, requires_grad=True),
            torch.randn(3, 128, 128, requires_grad=True)
        ]

        relu = torch.nn.ReLU()
        relu_ = torch.nn.ReLU(inplace=True)
        tensor_res = []
        for i in range(2):
            t_res = relu(inputs[i].unsqueeze(0).contiguous())
            t_res = relu_(t_res)
            tensor_res.append(t_res.squeeze(0))
            tensor_res[i].sum().backward()
        layer_grad0 = [p.grad for (n, p) in relu.named_parameters()]

        nt = ntnt(inputs)
        nt_res = relu(nt)
        nt_res = relu_(nt_res)
        nt_res.sum().backward()
        layer_grad1 = [p.grad for (n, p) in relu.named_parameters()]

        self.assertEqual(ntnt(tensor_res), nt_res)
        map(self.assertEqual, zip(layer_grad0, layer_grad1))
        self.assertEqual(inputs[0].grad, nt.grad[0])
        self.assertEqual(inputs[1].grad, nt.grad[1])

    @unittest.skip("Requires autograd support")
    def test_add(self):
        inputs0_ = [
            torch.randn(5, 6, requires_grad=True),
            torch.randn(1, 1, requires_grad=True)
        ]
        inputs1_ = [
            torch.randn(5, 6, requires_grad=True),
            torch.randn(1, 1, requires_grad=True)
        ]
        inputs0 = ntnt(inputs0_)
        inputs1 = ntnt(inputs1_)
        output = inputs0 + inputs1
        output += inputs0
        output.sum().backward()
        self.assertEqual(inputs0.grad.sum(),
                         inputs1.grad.sum() + inputs1.grad.sum())

    @unittest.skip("Requires autograd support")
    def test_resnet_bottleneck(self):
        import torchvision

        def _test(Bottleneck, has_grad=True):
            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = Bottleneck()
            print(b)
            x = b(inputs).sum()
            # import torchviz
            # dot = torchviz.make_dot(x)
            # dot.format = 'svg'
            # dot.render('asdf')
            # x.backward()
            # import sys; sys.exit(1)
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True),
                torch.randn(256, 18, 18, requires_grad=True)
            ]
            b = Bottleneck()
            inputs = ntnt(inputs_)
            if has_grad:
                b(inputs).sum().backward()
                # print(list((n, p.grad is None) for (n, p) in b.named_parameters()))

                b.zero_grad()
                b(inputs_[0].unsqueeze(0)).sum().backward()

                b.zero_grad()
                b(inputs_[1].unsqueeze(0)).sum().backward()

                self.assertEqual(inputs_[0].grad, inputs.grad[0])
                self.assertEqual(inputs_[1].grad, inputs.grad[1])
        _test(lambda: torchvision.models.resnet.Bottleneck(256, 64), False)
        _test(lambda: torchvision.models.resnet.Bottleneck(256, 64).eval())

    @unittest.skip("Requires autograd support")
    def test_resnet_classification(self):
        import torchvision

        def _test(FCNHead):
            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = FCNHead()
            print(b)
            # print(b)
            # list(b.children())[3].eval()  # dropout is stochastic otherwise
            b(inputs).sum().backward()
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(256, 50, 60, requires_grad=True),
                torch.randn(256, 18, 18, requires_grad=True)
            ]
            inputs = ntnt(inputs_)
            b.zero_grad()
            b(inputs).sum().backward()

            b.zero_grad()
            b(inputs_[0].unsqueeze(0)).sum().backward()

            b.zero_grad()
            b(inputs_[1].unsqueeze(0)).sum().backward()

            self.assertEqual(inputs_[0].grad, inputs.grad[0])
            self.assertEqual(inputs_[1].grad, inputs.grad[1])
        # _test(lambda: torchvision.models.segmentation.fcn.FCNHead(256, 64))
        _test(lambda: torchvision.models.segmentation.fcn.FCNHead(256, 64).eval())

    @unittest.skip("Requires autograd support")
    def test_backbone(self):
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter

        def _test(FCNHead):
            inputs_ = [
                torch.randn(3, 50, 60, requires_grad=True)
            ]
            inputs = ntnt(inputs_)

            b = FCNHead()
            # print(b)
            # print(b(inputs))
            b(inputs)[0][0].sum().backward()
            g0 = list(p.grad for (n, p) in b.named_parameters())

            b.zero_grad()
            b(inputs_[0].unsqueeze(0))[0][0].sum().backward()
            g1 = list(p.grad for (n, p) in b.named_parameters())

            map(self.assertEqual, zip(g0, g1))

            inputs_ = [
                torch.randn(3, 50, 60, requires_grad=True),
                torch.randn(3, 18, 18, requires_grad=True)
            ]
            inputs = ntnt(inputs_)
            b.zero_grad()
            b(inputs)[0][0].sum().backward()
            # for (n, p) in b.named_parameters():
            #     if p.grad is None:
            #         print(n)
            #         continue
            #     print(n, " is fine")

            b.zero_grad()
            b(inputs_[0].unsqueeze(0))[0][0].sum().backward()

            b.zero_grad()
            b(inputs_[1].unsqueeze(0))[0][0].sum().backward()

            self.assertEqual(inputs_[0].grad, inputs.grad[0])
            self.assertEqual(inputs_[1].grad, inputs.grad[1])
        # Note: It seems expected that layer0 has no gradients.
        return_layers = {"layer1": "0", "layer2": "1",
                         "layer3": "2", "layer4": "3"}
        _test(lambda: Joiner(IntermediateLayerGetter(torchvision.models.resnet50(
            replace_stride_with_dilation=[False, False, False],
            pretrained=True, norm_layer=NTFrozenBatchNorm2d), return_layers),
            PositionEmbeddingSine(128, normalize=True)))

    @unittest.skip("Requires autograd support")
    def test_nn_max_pool2d(self):
        data = [
            [
                torch.randn(3, 500, 600),
                torch.randn(3, 128, 128)
            ],
            [
                torch.randn(3, 500, 600),
                torch.randn(3, 500, 600)
            ],
        ]

        # with optional params
        maxPool2d = torch.nn.MaxPool2d(kernel_size=(
            3, 3), stride=2, padding=(1, 1), dilation=1, ceil_mode=False)
        for inputs in data:
            tensor_res = []
            for i in range(2):
                t_res = maxPool2d(inputs[i].unsqueeze(0).contiguous())
                tensor_res.append(t_res.squeeze(0))

            nt = ntnt(inputs)
            nt_res = maxPool2d(nt)
            self.assertEqual(ntnt(tensor_res), nt_res)

    @unittest.skip("Requires autograd support")
    def test_fzbn2d(self):
        class FrozenBatchNorm2d(torch.nn.Module):
            """
            BatchNorm2d where the batch statistics and the affine parameters are fixed.
            Copy-paste from torchvision.misc.ops with added eps before rqsrt,
            without which any other models than torchvision.models.resnet[18,34,50,101]
            produce nans.
            """

            def __init__(self, n):
                super(FrozenBatchNorm2d, self).__init__()
                self.register_buffer("weight", torch.ones(n))
                self.register_buffer("bias", torch.zeros(n))
                self.register_buffer("running_mean", torch.zeros(n))
                self.register_buffer("running_var", torch.ones(n))

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                num_batches_tracked_key = prefix + 'num_batches_tracked'
                if num_batches_tracked_key in state_dict:
                    del state_dict[num_batches_tracked_key]

                super(FrozenBatchNorm2d, self)._load_from_state_dict(
                    state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs)

            def forward(self, x):
                # move reshapes to the beginning
                # to make it fuser-friendly
                w = self.weight.reshape(-1, 1, 1)
                b = self.bias.reshape(-1, 1, 1)
                rv = self.running_var.reshape(-1, 1, 1)
                rm = self.running_mean.reshape(-1, 1, 1)
                eps = 1e-5
                scale = w * (rv + eps).rsqrt()
                bias = b - rm * scale
                # return (x * scale + bias)
                # return x
                # return (x * scale + bias)
                return x + bias

        b0 = FrozenBatchNorm2d(64)  # .cuda()
        random.seed(1010)
        torch.manual_seed(1310)
        RAND_INTS = [random.randint(100, 300) for _ in range(1)]
        tensors = [torch.rand(64, i, 256, requires_grad=True)
                   for i in RAND_INTS]
        nested_tensor = ntnt(tensors)
        # print(nested_tensor.nested_size())
        s0 = b0(nested_tensor).sum()
        s0.backward()

        b1 = FrozenBatchNorm2d(64)
        s1 = 0
        for t in tensors:
            s1 += b1(t).sum()
        s1.backward()
        self.assertEqual(s0, s1)
        for i in range(len(tensors)):
            self.assertEqual(nested_tensor.grad[i], tensors[i].grad)

        self.assertEqual(len((list(b0.named_parameters()))), 0)
        self.assertEqual(len((list(b1.named_parameters()))), 0)


if __name__ == "__main__":
    unittest.main()
