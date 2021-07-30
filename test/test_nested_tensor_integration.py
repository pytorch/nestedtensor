import torch
import nestedtensor
import unittest
from utils_test_case import TestCase

try:
    import classy_vision
    TEST_CLASSY_VISION=True
except ModuleNotFoundError:
    TEST_CLASSY_VISION=False


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)
def ntnt_nograd(x): return nestedtensor.nested_tensor(x, requires_grad=False)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


class TestIntegration(TestCase):
    def test_resnet18(self):
        import torchvision
        EXAMPLE_IMAGE_TENSORS = [torch.randn(3, 10, 10) for _ in range(3)]
        model = torchvision.models.resnet.resnet18(pretrained=True).eval()
        with torch.inference_mode():
            result_model_nt = model(ntnt_nograd(
                EXAMPLE_IMAGE_TENSORS)).unbind()
            result_model = model(torch.stack(EXAMPLE_IMAGE_TENSORS)).unbind()
        for t0, t1 in zip(result_model_nt, result_model):
            self.assertEqual(t0, t1)

        # non-regular shape smoke test
        EXAMPLE_IMAGE_TENSORS = [torch.randn(
            3, 100 * i, 100) for i in range(1, 4)]
        with torch.inference_mode():
            model(ntnt_nograd(EXAMPLE_IMAGE_TENSORS))

    def test_segmentation_pretrained_test_only(self):

        def _test(seed, model_factory, use_confmat, num_classes=21):
            torch.manual_seed(seed)
            t1 = torch.randn(3, 3, 4, requires_grad=True)
            t2 = torch.randn(3, 3, 4, requires_grad=True)
            tr1 = torch.randn(3, 4, requires_grad=True)
            tr2 = torch.randn(3, 4, requires_grad=True)

            model1 = model_factory()
            # model1 = torchvision.models.segmentation.__dict__[model_name](
            #     num_classes=num_classes, aux_loss=aux_loss, pretrained=True
            # )
            # model1.eval()

            # tensor run
            t_input = torch.stack([t1, t2])
            t_target = torch.stack([tr1, tr2])
            if use_confmat:
                confmat = ConfusionMatrix(num_classes)

            output1 = model1(t_input)
            if use_confmat:
                output1 = output1["out"]
            else:
                output1 = output1["0"]

            if use_confmat:
                confmat.update(t_target.flatten(), output1.argmax(1).flatten())
                confmat.reduce_from_all_processes()

            # nt run
            # model2 = torchvision.models.segmentation.__dict__[model_name](
            #     num_classes=num_classes, aux_loss=aux_loss, pretrained=True
            # )
            # model2.eval()
            model2 = model_factory()
            nt_t1 = t1.clone().detach()
            nt_t2 = t2.clone().detach()
            nt_tr1 = tr1.clone().detach()
            nt_tr2 = tr2.clone().detach()

            nt_input = ntnt_nograd(
                [nt_t1, nt_t2])
            nt_target = ntnt_nograd(
                [nt_tr1, nt_tr2])
            if use_confmat:
                confmat2 = ConfusionMatrix(num_classes)

            with torch.inference_mode():
                output2 = model2(nt_input)
            if use_confmat:
                output2 = output2["out"]
            else:
                output2 = output2["0"]

            if use_confmat:
                for a, b in zip(nt_target, output2):
                    confmat2.update(a.flatten(), b.argmax(0).flatten())

            if use_confmat:
                confmat2.reduce_from_all_processes()
                self.assertEqual(confmat.mat, confmat2.mat)

            # grad test
            self.assertEqual(ntnt_nograd(output1.unbind()), output2)

            # output1.sum().backward()
            # output2.sum().backward()

            # for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            #     if p1.grad is not None:
            #         self.assertEqual(p1.grad, p2.grad)
            #     else:
            #         self.assertIsNone(p2.grad)

            # # TODO: Re-enable under autograd support
            # self.assertEqual(t1.grad, nt_input.grad[0])
            # self.assertEqual(t2.grad, nt_input.grad[1])

        import torchvision
        _test(10, lambda: torchvision.models.segmentation.__dict__["fcn_resnet101"](
            num_classes=21, aux_loss="store_true", pretrained=True
        ).eval(), True)

        # _test(1010, lambda: IntermediateLayerGetter(getattr(torchvision.models, "resnet18")(
        #     replace_stride_with_dilation=[False, False, False],
        #     pretrained=True, norm_layer=NTFrozenBatchNorm2d), {'layer4': "0"}), False)

    def test_transformer_forward(self):
        EMBED_DIM = 32
        NHEAD = 8
        t = torch.nn.Transformer(EMBED_DIM, NHEAD, dropout=0.0)

        src0 = torch.randn(2, EMBED_DIM)
        src1 = torch.randn(4, EMBED_DIM)
        nt_src = ntnt_nograd([src0, src1])

        tgt0 = torch.randn(3, EMBED_DIM)
        tgt1 = torch.randn(5, EMBED_DIM)
        nt_tgt = ntnt_nograd([tgt0, tgt1])

        res_0 = t(src0.unsqueeze(1), tgt0.unsqueeze(1)).squeeze(1)
        res_1 = t(src1.unsqueeze(1), tgt1.unsqueeze(1)).squeeze(1)
        with torch.inference_mode():
            res_nt = t(nt_src, nt_tgt)

        for t0, t1 in zip(res_nt.unbind(), [res_0, res_1]):
            self.assertEqual(t0, t1)

    @unittest.skipIf(not TEST_CLASSY_VISION, "No classy vision")
    def test_fusion_resnext101_32x4d(self):
        @torch.inference_mode()
        def _test(dtype, use_channels_last):
            from classy_vision.models import build_model
            from torch.fx import symbolic_trace
            model = build_model({"name": "resnext101_32x4d"}).eval().cuda()
            model._initialize_weights(False)
            fused = symbolic_trace(model)
            fused = nestedtensor.fuse_conv_bn(fused)
            fused = nestedtensor.fuse_conv_relu(fused)
            model = model.to(dtype)
            fused = fused.to(dtype)
            data = torch.randn(2, 3, 50, 50, device=torch.device('cuda'), dtype=dtype)
            if use_channels_last:
                data = data.contiguous(memory_format=torch.channels_last)
            ref_output = model(data)
            new_output = fused(data)
            if dtype == torch.float16:
                self.assertEqual(ref_output, new_output, prec=2e-3)
            else:
                self.assertEqual(ref_output, new_output)
        _test(torch.float16, False)
        _test(torch.float32, False)
        # _test(torch.float16, True)
        _test(torch.float32, True)


if __name__ == "__main__":
    unittest.main()
