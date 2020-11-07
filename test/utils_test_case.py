import unittest
import numpy
import torch
import nestedtensor as NT
from numbers import Number
from math import inf
from collections import OrderedDict

string_classes = (str, bytes)

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

# NOTE: Methods copy pasted from https://github.com/pytorch/pytorch/blob/4314620ba05bc1867f6a63455c4ac77fdfb1018d/test/common_utils.py#L773
class TestCaseBase(unittest.TestCase):
    longMessage = True
    precision = 1e-5

    def safeCoalesce(self, t):
        tc = t.coalesce()
        self.assertEqual(tc.to_dense(), t.to_dense())
        self.assertTrue(tc.is_coalesced())

        # Our code below doesn't work when nnz is 0, because
        # then it's a 0D tensor, not a 2D tensor.
        if t._nnz() == 0:
            self.assertEqual(t._indices(), tc._indices())
            self.assertEqual(t._values(), tc._values())
            return tc

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx.tolist())
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map[idx] for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)

        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())

        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())

        if t.is_coalesced():
            self.assertEqual(tc._indices(), t._indices())
            self.assertEqual(tc._values(), t._values())

        return tg

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, numpy.bool_):
            self.assertEqual(x.item(), y, prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, numpy.bool_):
            self.assertEqual(x, y.item(), prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            def assertTensorsEqual(a, b):
                super(TestCaseBase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    if (a.device.type == 'cpu' and (a.dtype == torch.float16 or a.dtype == torch.bfloat16)):
                        # CPU half and bfloat16 tensors don't have the methods we need below
                        a = a.to(torch.float32)
                    if (a.device.type == 'cuda' and a.dtype == torch.bfloat16):
                        # CUDA bfloat16 tensors don't have the methods we need below
                        a = a.to(torch.float32)
                    b = b.to(a)

                    if (a.dtype == torch.bool) != (b.dtype == torch.bool):
                        raise TypeError("Was expecting both tensors to be bool type.")
                    else:
                        if a.dtype == torch.bool and b.dtype == torch.bool:
                            # we want to respect precision but as bool doesn't support subtraction,
                            # boolean tensor has to be converted to int
                            a = a.to(torch.int)
                            b = b.to(torch.int)

                        diff = a - b
                        if a.is_floating_point():
                            # check that NaNs are in the same locations
                            nan_mask = torch.isnan(a)
                            self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                            diff[nan_mask] = 0
                            # inf check if allow_inf=True
                            if allow_inf:
                                inf_mask = torch.isinf(a)
                                inf_sign = inf_mask.sign()
                                self.assertTrue(torch.equal(inf_sign, torch.isinf(b).sign()), message)
                                diff[inf_mask] = 0
                        # TODO: implement abs on CharTensor (int8)
                        if diff.is_signed() and diff.dtype != torch.int8:
                            diff = diff.abs()
                        max_err = diff.max()
                        self.assertLessEqual(max_err, prec, message)
            super(TestCaseBase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            super(TestCaseBase, self).assertEqual(x.is_quantized, y.is_quantized, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            elif x.is_quantized and y.is_quantized:
                self.assertEqual(x.qscheme(), y.qscheme(), prec=prec,
                                 message=message, allow_inf=allow_inf)
                if x.qscheme() == torch.per_tensor_affine:
                    self.assertEqual(x.q_scale(), y.q_scale(), prec=prec,
                                     message=message, allow_inf=allow_inf)
                    self.assertEqual(x.q_zero_point(), y.q_zero_point(),
                                     prec=prec, message=message,
                                     allow_inf=allow_inf)
                elif x.qscheme() == torch.per_channel_affine:
                    self.assertEqual(x.q_per_channel_scales(), y.q_per_channel_scales(), prec=prec,
                                     message=message, allow_inf=allow_inf)
                    self.assertEqual(x.q_per_channel_zero_points(), y.q_per_channel_zero_points(),
                                     prec=prec, message=message,
                                     allow_inf=allow_inf)
                    self.assertEqual(x.q_per_channel_axis(), y.q_per_channel_axis(),
                                     prec=prec, message=message)
                self.assertEqual(x.dtype, y.dtype)
                self.assertEqual(x.int_repr().to(torch.int32),
                                 y.int_repr().to(torch.int32), prec=prec,
                                 message=message, allow_inf=allow_inf)
            else:
                assertTensorsEqual(x, y)
        elif isinstance(x, string_classes) and isinstance(y, string_classes):
            super(TestCaseBase, self).assertEqual(x, y, message)
        elif type(x) == set and type(y) == set:
            super(TestCaseBase, self).assertEqual(x, y, message)
        elif isinstance(x, dict) and isinstance(y, dict):
            if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                self.assertEqual(x.items(), y.items(), prec=prec,
                                 message=message, allow_inf=allow_inf)
            else:
                self.assertEqual(set(x.keys()), set(y.keys()), prec=prec,
                                 message=message, allow_inf=allow_inf)
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list],
                                 [y[k] for k in key_list],
                                 prec=prec, message=message,
                                 allow_inf=allow_inf)
        elif is_iterable(x) and is_iterable(y):
            super(TestCaseBase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec=prec, message=message,
                                 allow_inf=allow_inf)
        elif isinstance(x, bool) and isinstance(y, bool):
            super(TestCaseBase, self).assertEqual(x, y, message)
        elif isinstance(x, Number) and isinstance(y, Number):
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCaseBase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCaseBase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCaseBase, self).assertEqual(x, y, message)

    def assertAlmostEqual(self, x, y, places=None, msg=None, delta=None, allow_inf=None):
        prec = delta
        if places:
            prec = 10**(-places)
        self.assertEqual(x, y, prec, msg, allow_inf)

    def assertNotEqual(self, x, y, prec=None, message=''):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.size() != y.size():
                super(TestCaseBase, self).assertNotEqual(x.size(), y.size())
            self.assertGreater(x.numel(), 0)
            y = y.type_as(x)
            y = y.cuda(device=x.get_device()) if x.is_cuda else y.cpu()
            nan_mask = x != x
            if torch.equal(nan_mask, y != y):
                diff = x - y
                if diff.is_signed():
                    diff = diff.abs()
                diff[nan_mask] = 0
                # Use `item()` to work around:
                # https://github.com/pytorch/pytorch/issues/22301
                max_err = diff.max().item()
                self.assertGreaterEqual(max_err, prec, message)
        elif type(x) == str and type(y) == str:
            super(TestCaseBase, self).assertNotEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCaseBase, self).assertNotEqual(x, y)
        else:
            try:
                self.assertGreaterEqual(abs(x - y), prec, message)
                return
            except (TypeError, AssertionError):
                pass
            super(TestCaseBase, self).assertNotEqual(x, y, message)

class TestCase(TestCaseBase):
    # ToDo: remove ignore_contiguity flag. We should not use it.
    def assertAlmostEqual(self, x, y, places=None, msg=None, delta=None, allow_inf=None, ignore_contiguity=False):
        prec = delta
        if places:
            prec = 10**(-places)
        self.assertEqual(x, y, prec, msg, allow_inf, ignore_contiguity)

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False, ignore_contiguity=False):
        if not isinstance(x, NT.NestedTensor) and not isinstance(y, NT.NestedTensor):
            super().assertEqual(x, y, prec, message, allow_inf)
        elif not isinstance(x, NT.NestedTensor) or not isinstance(y, NT.NestedTensor):
            raise TypeError("Comparing a nested tensor to a non nested tensor")
        else:
            if x.dim() != y.dim():
                self.fail("Nested tensors dimensionality don't match. {} != {}".format(x.dim(), y.dim()))

            if x.nested_dim() != y.nested_dim():
                self.fail("Nested tensors nested dimensionality don't match. {} != {}".format(x.nested_dim(), y.nested_dim()))

            if x.tensor_dim() != y.tensor_dim():
                self.fail("Nested tensors  dimentionality don't match. {} != {}".format(x.tensor_dim(), y.tensor_dim()))

            if x.is_pinned() != y.is_pinned():
                self.fail("Nested tensors pinned memmory values don't match. {} != {}".format(x.is_pinned(), y.is_pinned()))

            if x.layout != y.layout:
                self.fail("Nested tensors layouts don't match. {} != {}".format(x.layout, y.layout))

            if x.dtype != y.dtype:
                self.fail("Nested tensors dtypes don't match. {} != {}".format(x.dtype, y.dtype))

            if x.device != y.device:
                self.fail("Nested tensors devices don't match. {} != {}".format(x.device, y.device))

            if x.requires_grad != y.requires_grad:
                self.fail("Nested tensors requires grad properties don't match. {} != {}".format(x.requires_grad, y.requires_grad))

            # uncomment once nested_tensor([]).is_contiguous() == nested_tensor([], dtype=torch.float).is_contiguous()
            #if not ignore_contiguity and x.is_contiguous() != y.is_contiguous():
            #    self.fail("Nested tensors contiguity don't match. {} != {}".format(x.is_contiguous(), y.is_contiguous()))

            if x.element_size() != y.element_size():
                self.fail("Nested tensors element sizes don't match. {} != {}".format(x.element_size(), y.element_size()))

            if x.size() != y.size():
                self.fail("Nested tensors sizes don't match. {} != {}".format(x.size(), y.size()))

            if x.nested_size() != y.nested_size():
                print(x.nested_size())
                print(y.nested_size())
                self.fail("Nested tensors nested sizes don't match. {} != {}".format(x.nested_size(), y.nested_size()))

            # If you ignore contiguity you should also ignore the striding
            if not ignore_contiguity and x.nested_stride() != y.nested_stride():
                self.fail("Nested tensors nested strides don't match. {} != {}".format(x.nested_stride(), y.nested_stride()))

            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec=prec, message=message,
                                 allow_inf=allow_inf, ignore_contiguity=ignore_contiguity)
