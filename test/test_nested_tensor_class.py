import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from utils import TestCase
import random

import utils


def ntnt(x): return nestedtensor.nested_tensor(x, requires_grad=True)
def ntnt_nograd(x): return nestedtensor.nested_tensor(x)

# Given arguments to a constructor iterator over results for
# as_nested_tensor and nested_tensor constructors.


def _iter_constructors():
    yield nestedtensor.as_nested_tensor
    yield nestedtensor.nested_tensor


def _test_property(self, fn):
    for constructor in _iter_constructors():
        num_nested_tensor = 3
        nested_tensor_lists = [utils.gen_nested_list(i, i, 3)
                               for i in range(1, num_nested_tensor)]
        first_tensors = [utils.get_first_tensor(
            ntl) for ntl in nested_tensor_lists]
        nested_tensors = [constructor(ntl) for ntl in nested_tensor_lists]
        for nested_tensor, first_tensor in zip(nested_tensors, first_tensors):
            self.assertEqual(fn(nested_tensor), fn(first_tensor))


class TestNestedTensor(TestCase):

    def test_nested_constructor(self):
        for constructor in _iter_constructors():
            num_nested_tensor = 3
            # TODO: Shouldn't be constructable
            [utils.gen_nested_tensor(i, i, 3, constructor=constructor)
             for i in range(1, num_nested_tensor)]

    def test_list_constructor(self):
        """
        This tests whether nestedtensor.as_nested_tensor stores Variables that share storage with
        the input Variables used for construction.
        """
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(utils.gen_float_tensor(i, (i + 1, 128, 128)))
        nested_tensor = nestedtensor.as_nested_tensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertNotEqual(tensors[i], nested_tensor.unbind()[i])
            self.assertNotEqual(tensors[i].storage().data_ptr(
            ), nested_tensor.unbind()[i].storage().data_ptr())

    def test_as_nested_tensor(self):
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(utils.gen_float_tensor(i, (i + 1, 128, 128)))

        # This should NOT create references
        nested_tensor = nestedtensor.as_nested_tensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertNotEqual(tensors[i], nested_tensor.unbind()[i])

        # This should NOT create references
        nested_tensor = nestedtensor.nested_tensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertNotEqual(tensors[i], nested_tensor.unbind()[i])

        nested_tensor1 = nestedtensor.as_nested_tensor(nested_tensor)
        self.assertTrue(nested_tensor1 is nested_tensor)
        self.assertRaises(NotImplementedError, lambda: nestedtensor.as_nested_tensor(
            nested_tensor, dtype=torch.int64))
        # self.assertTrue(nested_tensor2 is not nested_tensor)

    def test_constructor(self):
        for constructor in _iter_constructors():
            self.assertRaises(
                RuntimeError, lambda: constructor([3.0]))
            self.assertRaises(
                TypeError, lambda: constructor(torch.tensor([3.0])))
            for constructor2 in _iter_constructors():
                constructor(
                    constructor2([torch.tensor([3.0])]))
            for constructor2 in _iter_constructors():
                self.assertRaises(RuntimeError, lambda: constructor(
                    [torch.tensor([2.0]), constructor2([torch.tensor([3.0])])]))
            self.assertRaises(TypeError, lambda: constructor(4.0))

    def test_default_constructor(self):
        # nested_dim is 1 and dim is 1 too.
        for constructor in _iter_constructors():
            self.assertRaises(TypeError, lambda: constructor())
            default_nested_tensor = constructor([])
            default_tensor = torch.tensor([])
            self.assertEqual(default_nested_tensor.nested_dim(), 1)
            self.assertEqual(default_nested_tensor.nested_size().unbind(), [])
            self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
            self.assertEqual(default_nested_tensor.layout,
                             default_tensor.layout)
            self.assertEqual(default_nested_tensor.device,
                             default_tensor.device)
            self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
            self.assertEqual(default_nested_tensor.requires_grad,
                             default_tensor.requires_grad)
            self.assertIsNone(default_tensor.grad)
            self.assertEqual(default_nested_tensor.is_pinned(),
                             default_tensor.is_pinned())

    # def test_scalar_constructor(self):
    #     # Not a valid NestedTensor. This is not a list of Tensors or constructables for Tensors.
    #     ntimeError, lambda: nestedtensor.nested_tensor([1.0]))

    def test_repr_string(self):
        for constructor in _iter_constructors():
            a = constructor(
                [
                ])
            expected = "nested_tensor(["\
                       "\n\n])"
            self.assertEqual(str(a), expected)
            self.assertEqual(repr(a), expected)

            a = constructor(
                [
                    torch.tensor(1),
                ])
            expected = "nested_tensor(["\
                       "\n\ttensor(1)"\
                       "\n])"
            # self.assertEqual(str(a), expected)
            # self.assertEqual(repr(a), expected)
            str(a)
            repr(a)

            a = constructor(
                [
                    torch.tensor([[1, 2]]),
                    torch.tensor([[4, 5]]),
                ])
            expected = "nested_tensor(["\
                       "\n\ttensor([[1, 2]])"\
                       ","\
                       "\n\ttensor([[4, 5]])"\
                       "\n])"
            # self.assertEqual(str(a), expected)
            # self.assertEqual(repr(a), expected)
            str(a)
            repr(a)

            a = constructor(
                [
                    [torch.tensor([[1, 2], [2, 3]]), torch.tensor([[3, 4]])],
                    [torch.tensor([[4, 5]])]
                ])
            expected = "nested_tensor(["\
                       "\n\tnested_tensor(["\
                       "\n\t\ttensor([[1, 2]"\
                       ","\
                       "\n\t\t        [2, 3]])"\
                       ","\
                       "\n\t\ttensor([[3, 4]])"\
                       "\n\t])"\
                       ","\
                       "\n\tnested_tensor(["\
                       "\n\t\ttensor([[4, 5]])"\
                       "\n\t])"\
                       "\n])"
            # self.assertEqual(str(a), expected)
            # self.assertEqual(repr(a), expected)
            str(a)
            repr(a)

    def test_element_size(self):
        for constructor in _iter_constructors():
            nt1 = constructor([])
            self.assertEqual(nt1.element_size(), torch.randn(1).element_size())
            a = torch.randn(4).int()
            nt2 = constructor([a])
            self.assertEqual(a.element_size(), nt2.element_size())

    def test_nested_size(self):
        for constructor in _iter_constructors():
            a = constructor([])
            self.assertEqual(len(a.nested_size()), 0)
            self.assertRaises(RuntimeError, lambda: a.nested_size()[0])

            a = constructor([torch.tensor(1)])
            self.assertEqual(len(a.nested_size()), 1)
            self.assertEqual(a.nested_size()[0], torch.Size([]))
            self.assertEqual(a.nested_size(0), 1)
            self.assertRaises(IndexError, lambda: a.nested_size(1))

            a = constructor([torch.randn(1)])
            self.assertEqual(a.nested_size()[0], torch.Size([1]))
            self.assertEqual(a.nested_size()[0][0], 1)
            self.assertEqual(a.nested_size(0), 1)
            self.assertEqual(a.nested_size(1), (1,))
            self.assertRaises(IndexError, lambda: a.nested_size(2))

            a = constructor([torch.randn(1, 2)])
            self.assertEqual(a.nested_size()[0], torch.Size([1, 2]))
            self.assertEqual(a.nested_size(0), 1)
            self.assertEqual(a.nested_size(1), (1,))
            self.assertEqual(a.nested_size(2), (2,))
            self.assertRaises(IndexError, lambda: a.nested_size(3))

            # Make sure object is not bound to life-time of NestedTensor instance
            b = a.nested_size()
            del a
            self.assertEqual(len(b), 1)
            self.assertEqual(b[0], torch.Size([1, 2]))
            self.assertEqual(b[0][0], 1)
            self.assertEqual(b[0][1], 2)

            a = constructor(
                [[torch.randn(1)], [torch.randn(2), torch.randn(1)]])
            self.assertEqual(a.nested_size()[0][0], torch.Size([1]))
            self.assertEqual(a.nested_size()[1][0], torch.Size([2]))
            self.assertEqual(a.nested_size()[1][1], torch.Size([1]))
            self.assertEqual(a.nested_size(0), 2)
            self.assertEqual(a.nested_size(1), (1, 2))
            self.assertEqual(a.nested_size(2), ((1,), (2, 1)))
            self.assertRaises(IndexError, lambda: a.nested_size(3))

            a = constructor([[torch.tensor(1)],
                             [torch.tensor(2), torch.tensor(1)]])
            self.assertEqual(a.nested_size()[0][0], torch.Size([]))
            self.assertEqual(a.nested_size()[1][0], torch.Size([]))
            self.assertEqual(a.nested_size()[1][1], torch.Size([]))
            self.assertEqual(a.nested_size(0), 2)
            self.assertEqual(a.nested_size(1), (1, 2))
            self.assertRaises(IndexError, lambda: a.nested_size(2))

    def test_nested_stride(self):
        for constructor in _iter_constructors():
            tensors = [torch.rand(1, 2, 4)[:, :, 0], torch.rand(
                2, 3, 4)[:, 1, :], torch.rand(3, 4, 5)[1, :, :]]
            a = constructor(tensors)
            na = list(list(t.contiguous().stride()) for t in tensors)
            ans = a.nested_stride()
            result = tuple(ans[i] for i in range(len(ans)))
            for r, s in zip(result, na):
                self.assertEqual(r, s)

    def test_len(self):
        for constructor in _iter_constructors():
            a = constructor([torch.tensor([1, 2]),
                             torch.tensor([3, 4]),
                             torch.tensor([5, 6]),
                             torch.tensor([7, 8])])
            self.assertEqual(len(a), 4)
            a = constructor([torch.tensor([1, 2]),
                             torch.tensor([7, 8])])
            self.assertEqual(len(a), 2)
            a = constructor([torch.tensor([1, 2])])
            self.assertEqual(len(a), 1)

    def test_equal(self):
        for constructor in _iter_constructors():
            a1 = constructor([torch.tensor([1, 2]),
                              torch.tensor([7, 8])])
            a2 = constructor([torch.tensor([1, 2]),
                              torch.tensor([7, 8])])
            a3 = constructor([torch.tensor([3, 4]),
                              torch.tensor([5, 6])])
            self.assertTrue((a1 == a2).all())
            self.assertTrue((a1 != a3).all())
            self.assertTrue(not (a1 != a2).any())
            self.assertTrue(not (a1 == a3).any())

            a1 = constructor([torch.tensor([1, 2]),
                              torch.tensor([2, 8])])
            if constructor == nestedtensor.as_nested_tensor:
                self.assertRaises(NotImplementedError, lambda: constructor([torch.tensor([0, 1]),
                                                                            torch.tensor([1, 0])], dtype=torch.bool))
                self.assertRaises(NotImplementedError, lambda: constructor([torch.tensor([1, 0]),
                                                                            torch.tensor([0, 1])], dtype=torch.bool))
            else:
                a2 = constructor([torch.tensor([0, 1]),
                                  torch.tensor([1, 0])], dtype=torch.bool)
                a3 = constructor([torch.tensor([1, 0]),
                                  torch.tensor([0, 1])], dtype=torch.bool)
                self.assertEqual((a1 == 2), a2)
                self.assertEqual((a1 != 2), a3)
                self.assertEqual((a1 == 2.0), a2)
                self.assertEqual((a1 != 2.0), a3)

    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)
            a1 = constructor([
                [torch.tensor([1, 2, 3, 4])],
                [torch.tensor([5, 6, 7, 8]), torch.tensor([9, 0, 0, 0])]
            ])
            self.assertEqual(a1.dim(), 3)

    def test_nested_dim(self):
        for constructor in _iter_constructors():
            nt = constructor([torch.tensor(3)])
            for i in range(2, 5):
                nt = utils.gen_nested_tensor(i, i, 3, constructor=constructor)
                self.assertEqual(nt.nested_dim(), i)

    def test_unbind(self):
        # This is the most important operation. We want to make sure
        # that the Tensors we use for construction can be retrieved
        # and used independently while still being kept track of.

        # In fact nestedtensor.as_nested_tensor behaves just like a list. Any
        # list of torch.Tensors you initialize it with will be
        # unbound to have the same id. That is, they are indeed
        # the same Variable, since each torch::autograd::Variable has
        # assigned to it a unique PyObject* by construction.

        # TODO: Check that unbind returns torch.Tensors when nested_dim is 1
        # TODO: contiguous nestedtensors should return tuples of contiguous nestedtensors on dimension 0

        def _test_fn(unbind_fn):
            def _test(a, b, c, d, e):
                nt = nestedtensor.nested_tensor([a, b])
                a1, b1 = nt.unbind()
                self.assertTrue(a is not a1)
                self.assertTrue(b is not b1)

                nt1 = nestedtensor.nested_tensor([[c, d], [e]])
                nt11, nt12 = unbind_fn(nt1, 0)
                c1, d1 = unbind_fn(nt11, 0)
                e1 = unbind_fn(nt12, 0)[0]

                self.assertTrue(c is not c1)
                self.assertTrue(d is not d1)
                self.assertTrue(e is not e1)

                nt = nestedtensor.nested_tensor([a, b])
                a1, b1 = unbind_fn(nt, 0)
                self.assertEqual(a, a1)
                self.assertEqual(b, b1)

                a = utils.gen_float_tensor(1, (2, 3)).add_(1)
                nt = nestedtensor.nested_tensor([a])
                self.assertEqual(a, unbind_fn(nt, 0)[0])

            _test(torch.tensor([1, 2]),
                  torch.tensor([7, 8]),
                  torch.tensor([3, 4]),
                  torch.tensor([5, 6]),
                  torch.tensor([6, 7]))
            _test(torch.tensor([1]),
                  torch.tensor([7]),
                  torch.tensor([3]),
                  torch.tensor([5]),
                  torch.tensor([6]))
            _test(torch.tensor(1),
                  torch.tensor(7),
                  torch.tensor(3),
                  torch.tensor(5),
                  torch.tensor(6))
            _test(torch.tensor([]),
                  torch.tensor([]),
                  torch.tensor([]),
                  torch.tensor([]),
                  torch.tensor([]))
        _test_fn(lambda x, dim: x.unbind(dim))
        _test_fn(lambda x, dim: torch.unbind(x, dim))

    def test_unbind_dim(self):
        # Unbinding across nested dimensions or tensors dimensions
        # is akin splitting up the tree across a level.

        def _test_fn(unbind_fn):
            # nt = nestedtensor.nested_tensor([])
            # self.assertEqual(unbind_fn(nt, 0), ())
            # self.assertRaises(IndexError, lambda: unbind_fn(nt, 1))

            a = torch.rand(3, 2)
            nt = nestedtensor.nested_tensor([a])
            # self.assertEqual(unbind_fn(nt, 0), (a,))
            result = (
                nestedtensor.nested_tensor([unbind_fn(a, 0)[0]]),
                nestedtensor.nested_tensor([unbind_fn(a, 0)[1]]),
                nestedtensor.nested_tensor([unbind_fn(a, 0)[2]]))
            # print('unbind_fn: ', unbind_fn)
            for x, y in zip(unbind_fn(nt, 1), result):
                # print('x: ', type(x), ' - y: ', type(y))
                self.assertEqual(x, y, ignore_contiguity=True)
            result = (
                nestedtensor.nested_tensor([unbind_fn(a, 1)[0]]),
                nestedtensor.nested_tensor([unbind_fn(a, 1)[1]]))
            for x, y in zip(unbind_fn(nt, 2), result):
                self.assertEqual(x, y, ignore_contiguity=True)

            b = torch.rand(2, 3)
            nt = nestedtensor.nested_tensor([a, b])
            self.assertEqual(unbind_fn(nt, 0), (a, b))
            result = (
                nestedtensor.nested_tensor(
                    [unbind_fn(a, 0)[0], unbind_fn(b, 0)[0]]),
                nestedtensor.nested_tensor(
                    [unbind_fn(a, 0)[1], unbind_fn(b, 0)[1]]),
                nestedtensor.nested_tensor([unbind_fn(a, 0)[2]]))
            for x, y in zip(unbind_fn(nt, 1), result):
                self.assertEqual(x, y, ignore_contiguity=True)
            # TODO: Add more tensors and unbind across more dimensions to create mixing

            c = torch.rand(4, 3)
            nt = nestedtensor.nested_tensor([[a], [b, c]])
            nt_a, nt_b = unbind_fn(nt, 0)
            self.assertEqual(nt_a, nestedtensor.nested_tensor(
                [a]), ignore_contiguity=True)
            self.assertEqual(nt_b, nestedtensor.nested_tensor(
                [b, c]), ignore_contiguity=True)
            result = (
                nestedtensor.nested_tensor([a, b]),
                nestedtensor.nested_tensor([c]))
            for x, y in zip(unbind_fn(nt, 1), result):
                self.assertEqual(x, y, ignore_contiguity=True)
        _test_fn(lambda x, dim: x.unbind(dim))
        _test_fn(lambda x, dim: torch.unbind(x, dim))

    def test_size(self):
        for constructor in _iter_constructors():
            a = constructor([])
            self.assertEqual(a.size(), (0,))

            a = constructor([torch.tensor(1)])
            self.assertEqual(a.size(), (1,))

            a = constructor([torch.tensor(1), torch.tensor(2)])
            self.assertEqual(a.size(), (2,))

            a = constructor([[torch.rand(1, 8),
                              torch.rand(3, 8)],
                             [torch.rand(7, 8)]])
            self.assertEqual(a.size(), (2, None, None, 8))

            a = constructor([torch.rand(1, 2),
                             torch.rand(1, 8)])
            self.assertEqual(a.size(), (2, 1, None))

            a = constructor([torch.rand(3, 4),
                             torch.rand(5, 4)])
            self.assertEqual(a.size(), (2, None, 4))

    def test_to_tensor(self):
        for constructor in _iter_constructors():
            a = constructor([])
            self.assertEqual(a.to_tensor(0), torch.tensor([]))
            self.assertRaises(IndexError, lambda: a.to_tensor(1))
            self.assertRaises(IndexError, lambda: a.to_tensor(2))

            a = constructor([torch.tensor(1)])
            self.assertEqual(a.to_tensor(0), torch.tensor([1]))
            self.assertRaises(IndexError, lambda: a.to_tensor(1))
            self.assertRaises(IndexError, lambda: a.to_tensor(2))

            t_a = torch.randn(2, 3)
            t_b = torch.randn(2, 3)
            a = constructor([[t_a, t_b]])
            result = torch.stack([torch.stack([t_a, t_b])])
            self.assertEqual(a.to_tensor(), result)
            self.assertEqual(a.to_tensor(0), result)
            # self.assertEqual(a.to_tensor(1), nestedtensor.as_nested_tensor(
            #     [torch.stack([t_a, t_b])]))
            # self.assertEqual(a.to_tensor(
            #     2), nestedtensor.as_nested_tensor([[t_a, t_b]]))
            # self.assertEqual(a.to_tensor(
            #     3), nestedtensor.as_nested_tensor([[t_a, t_b]]))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(1))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(2))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(3))
            self.assertRaises(IndexError, lambda: a.to_tensor(4))

            t_c = torch.randn(2, 3)
            t_d = torch.randn(2, 3)
            a = constructor([[t_a, t_b], [t_c, t_d]])
            result = torch.stack(
                [torch.stack([t_a, t_b]), torch.stack([t_c, t_d])])
            self.assertEqual(a.to_tensor(), result)
            self.assertEqual(a.to_tensor(0), result)
            # self.assertEqual(a.to_tensor(1), nestedtensor.as_nested_tensor(
            #     [torch.stack([t_a, t_b]), torch.stack([t_c, t_d])]))
            # self.assertEqual(a.to_tensor(2), nestedtensor.as_nested_tensor(
            #     [[t_a, t_b], [t_c, t_d]]))
            # self.assertEqual(a.to_tensor(3), nestedtensor.as_nested_tensor(
            #     [[t_a, t_b], [t_c, t_d]]))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(1))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(2))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(3))
            self.assertRaises(IndexError, lambda: a.to_tensor(4))

            t_e = torch.randn(3, 2)
            t_f = torch.randn(3, 2)
            a = constructor([[t_a, t_b], [t_e, t_f]])
            self.assertRaises(IndexError, lambda: a.to_tensor(0))
            # self.assertEqual(a.to_tensor(1), nestedtensor.as_nested_tensor(
            #     [torch.stack([t_a, t_b]), torch.stack([t_e, t_f])]))
            # self.assertEqual(a.to_tensor(2), nestedtensor.as_nested_tensor(
            #     [[t_a, t_b], [t_e, t_f]]))
            # self.assertEqual(a.to_tensor(3), nestedtensor.as_nested_tensor(
            #     [[t_a, t_b], [t_e, t_f]]))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(1))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(2))
            self.assertRaises(RuntimeError, lambda: a.to_tensor(3))
            self.assertRaises(IndexError, lambda: a.to_tensor(4))

    def test_to_nested_tensor(self):
        for constructor in _iter_constructors():
            a = constructor([])
            self.assertEqual(a.to_nested_tensor(), constructor(
                []), ignore_contiguity=True)
            self.assertEqual(a.to_nested_tensor(
                0), constructor([]), ignore_contiguity=True)
            self.assertEqual(a, a.to_nested_tensor(1))
            self.assertRaises(IndexError, lambda: a.to_nested_tensor(2))

            a = constructor([torch.tensor(1)])
            self.assertEqual(a.to_nested_tensor(), constructor(
                [torch.tensor(1)]), ignore_contiguity=True)
            self.assertEqual(a.to_nested_tensor(0), constructor(
                [torch.tensor(1)]), ignore_contiguity=True)
            self.assertRaises(IndexError, lambda: a.to_nested_tensor(1))
            self.assertRaises(IndexError, lambda: a.to_nested_tensor(2))

            t_a = torch.randn(2, 3)
            t_b = torch.randn(3, 2)
            a = constructor([t_a, t_b])
            result = constructor([t_a, t_b])
            self.assertEqual(a.to_nested_tensor(), result)
            self.assertEqual(a.to_nested_tensor(0), result)
            result = constructor([t_a.unbind(0), t_b.unbind(0)])
            self.assertEqual(a.to_nested_tensor(1), result)
            result = constructor(
                [list(map(lambda x: x.unbind(), t_a.unbind())),
                 list(map(lambda x: x.unbind(), t_b.unbind()))]
            )
            self.assertEqual(a.to_nested_tensor(2), result)
            self.assertRaises(IndexError, lambda: a.to_nested_tensor(3))

            a = constructor([[t_a, t_b]])
            result = constructor([[t_a, t_b]])
            self.assertEqual(a.to_nested_tensor(), result)
            self.assertEqual(a.to_nested_tensor(0), result)
            self.assertEqual(a.to_nested_tensor(1), result)
            result = constructor([[t_a.unbind(0), t_b.unbind(0)]])
            self.assertEqual(a.to_nested_tensor(2), result)
            result = constructor([[list(map(lambda x: x.unbind(), t_a.unbind())),
                                   list(map(lambda x: x.unbind(), t_b.unbind()))]])
            self.assertEqual(a.to_nested_tensor(3), result)
            self.assertRaises(IndexError, lambda: a.to_nested_tensor(4))

            t_c = torch.randn(2, 4)
            a = constructor([[t_a, t_b], [t_c]])
            result = constructor([[t_a, t_b], [t_c]])
            self.assertEqual(a.to_nested_tensor(), result)
            self.assertEqual(a.to_nested_tensor(0), result)
            self.assertEqual(a.to_nested_tensor(1), result)
            result = constructor(
                [[t_a.unbind(), t_b.unbind()], [t_c.unbind()]])
            self.assertEqual(a.to_nested_tensor(2), result)
            result = constructor([[list(map(lambda x: x.unbind(), t_a.unbind())),
                                   list(map(lambda x: x.unbind(), t_b.unbind()))],
                                  [list(map(lambda x: x.unbind(), t_c.unbind()))]])
            self.assertEqual(a.to_nested_tensor(3), result)
            self.assertRaises(IndexError, lambda: a.to_nested_tensor(4))

            t = torch.randn(2, 3)
            self.assertEqual(t, nestedtensor.to_nested_tensor(t, 0))
            self.assertEqual(ntnt_nograd(t.unbind()),
                             nestedtensor.to_nested_tensor(t, 1))
            self.assertEqual(ntnt_nograd(
                [ti.unbind() for ti in t.unbind()]), nestedtensor.to_nested_tensor(t, 2))
            self.assertRaises(
                IndexError, lambda: nestedtensor.to_nested_tensor(t, 3))

    def test_to(self):
        tensors = [torch.randn(1, 8),
                   torch.randn(3, 8),
                   torch.randn(7, 8)]
        a1 = nestedtensor.nested_tensor(tensors)
        self.assertRaises(NotImplementedError, lambda: a1.to(torch.int64))
        # for a, b in zip(tensors, a2.unbind()):
        #     self.assertEqual(a.to(torch.int64), b)

    def test_dtype(self):
        _test_property(self, lambda x: x.dtype)

    def test_device(self):
        _test_property(self, lambda x: x.device)

    def test_layout(self):
        _test_property(self, lambda x: x.layout)

    @unittest.skip("Requires autograd support")
    def test_requires_grad(self):
        _test_property(self, lambda x: x.requires_grad)
        tensors = [torch.randn(1, 8),
                   torch.randn(3, 8),
                   torch.randn(7, 8)]
        a1 = nestedtensor.nested_tensor(tensors, requires_grad=True)
        self.assertIsNone(a1.grad)

    @unittest.skip("Requires autograd support")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not enabled.")
    def test_pin_memory(self):
        # Check if it can be applied widely
        nt = utils.gen_nested_tensor(1, 4, 3)
        nt1 = nt.pin_memory()

        # Make sure it's actually a copy
        self.assertFalse(nt.is_pinned())
        self.assertTrue(nt1.is_pinned())
        a1 = torch.randn(1, 2)
        a2 = torch.randn(2, 3)
        nt2 = nestedtensor.as_nested_tensor([a1, a2])
        self.assertFalse(a1.is_pinned())
        self.assertFalse(a2.is_pinned())

        # Double check property transfers
        nt3 = nt2.pin_memory()
        self.assertFalse(nt2.is_pinned())
        self.assertTrue(nt3.is_pinned())

        # Check whether pinned memory is applied to constiuents
        # and relevant constiuents only.
        a3, a4 = nt3.unbind()
        a5, a6 = nt2.unbind()
        self.assertFalse(a1.is_pinned())
        self.assertFalse(a2.is_pinned())
        self.assertTrue(a3.is_pinned())
        self.assertTrue(a4.is_pinned())
        self.assertFalse(a5.is_pinned())
        self.assertFalse(a6.is_pinned())

    def test_getitem(self):
        a, b, c = torch.randn(3, 4), torch.randn(4, 3), torch.randn(1, 3)
        nt = ntnt_nograd([[a, b], [c]])
        tmp = nt[0, :, 0]
        self.assertEqual(tmp[0], a[:, 0])
        self.assertEqual(tmp[1], b[:, 0])
        self.assertEqual(nt[0, :, 0].contiguous(),
                         ntnt_nograd([a[:, 0], b[:, 0]]))
        self.assertEqual(nt[None], ntnt_nograd([[[a, b], [c]]]))
        self.assertEqual(nt[0], ntnt_nograd([a, b]))  # Supports grad
        self.assertEqual(nt[:], nt)
        self.assertEqual(nt[:, 0], ntnt_nograd([a, c]))
        self.assertEqual(nt[-1:], ntnt_nograd([[c]]))
        self.assertEqual(nt[-1:, 0], ntnt_nograd([c]))
        self.assertEqual(nt[:, -1], ntnt_nograd([b, c]))
        self.assertEqual(nt[-1:, -1], ntnt_nograd([c]))
        self.assertEqual(nt[:, -1:], ntnt_nograd([[b], [c]]))
        self.assertEqual(nt[-1:, -1:], ntnt_nograd([[c]]))
        self.assertEqual(nt[:, -1:, None], ntnt_nograd([[b[None]], [c[None]]]))
        self.assertEqual(nt[-1:, :, None], ntnt_nograd([[c[None]]]))
        self.assertEqual(nt[:, 1:, None], ntnt_nograd([[b[None]], []]))
        nt = nestedtensor.nested_tensor([[a, b]])
        self.assertEqual(nt[0, 0], ntnt_nograd([a[0], b[0]]))
        self.assertEqual(nt[0, 1:], ntnt_nograd([a[1:], b[1:]]))
        self.assertEqual(nt[:1, :, 1:], ntnt_nograd([[a[1:], b[1:]]]))
        self.assertEqual(nt[:, :], nt)
        self.assertEqual(nt[:, None], ntnt_nograd([[[a, b]]]))
        self.assertRaisesRegex(IndexError,
                               "Dimension out of range \(expected to be in range of \[-1, 0\], but got 2\)",
                               lambda: nt[2])

    def test_cat(self):
        a = torch.arange(12).reshape(3, 4)
        b = a + 12
        c = b + 12

        nt0 = ntnt_nograd([a, b])
        nt1 = ntnt_nograd([c])
        self.assertEqual(torch.cat([nt0, nt1], dim=0), ntnt_nograd([a, b, c]))
        self.assertEqual(torch.cat(
            [nt0, nt1], dim=1), ntnt_nograd([torch.cat([a, c]), b]))
        self.assertEqual(torch.cat([nt0, nt1], dim=2), ntnt_nograd(
            [torch.cat([a, c], dim=1), b]))

    def test_stack(self):
        a = torch.arange(12).reshape(3, 4)
        b = a + 12
        c = b + 12

        nt0 = ntnt_nograd([a, b])
        nt1 = ntnt_nograd([c])
        self.assertEqual(torch.stack(
            [nt0, nt1], dim=0), ntnt_nograd([[a, b], [c]]))
        self.assertEqual(torch.stack(
            [nt0, nt1], dim=1),
            ntnt_nograd([torch.stack([a, c]), b.reshape(1, 3, 4)]))
        self.assertEqual(torch.stack(
            [nt0, nt1], dim=2),
            ntnt_nograd([torch.stack([a, c], dim=1), b.reshape(3, 1, 4)]))


class TestContiguous(TestCase):
    @unittest.skip("Requires autograd support")
    def test_contiguous(self):
        for _ in range(1, 10):
            # data = gen_nested_list(1, 2, 3, size_low=1, size_high=3)
            data = [[torch.rand(1, 2), torch.rand(3, 4)], [torch.rand(5, 6)]]
            nt = nestedtensor.nested_tensor(data)
            self.assertTrue(nt.is_contiguous())
            # buf = nt.flatten()
            self.assertEqual(nt, nt)
            a = nt + nt
        nt.cos_()
        nt.cos()

        a = nestedtensor.as_nested_tensor([torch.tensor([1, 2]),
                                           torch.tensor([3, 4]),
                                           torch.tensor([5, 6]),
                                           torch.tensor([7, 8])])
        self.assertTrue(a.is_contiguous())


if __name__ == "__main__":
    unittest.main()
