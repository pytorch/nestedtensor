import torch
import nestedtensor as nt
import unittest
from utils_test_case import TestCase


class TestTensorMask(TestCase):
    #
    # Group of tests to test to_tensor_mask()
    #
    def test_empty_nt(self):
        a = nt.nested_tensor([])
        tensor, mask = a.to_tensor_mask()

        TestCase.assertEqual(self,  mask, torch.tensor(False))
        TestCase.assertEqual(self,  tensor, torch.tensor([0]))

    # TODO once .to_list() bug fixed
    def test_empty_tensor(self):
        a = nt.nested_tensor([
               torch.tensor([])
            ])
        self.assertRaisesRegex(RuntimeError,
                "Empty tensors are not yet supported.",
                lambda: a.to_tensor_mask())

    def test_single_scalar(self):
        a = nt.nested_tensor([
            torch.tensor(1, dtype=torch.uint8)
        ])
        tensor, mask = a.to_tensor_mask()
        TestCase.assertEqual(
            self, tensor, torch.tensor([1], dtype=torch.uint8))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        TestCase.assertEqual(
            self, tensor, torch.tensor([1], dtype=torch.uint8))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        TestCase.assertEqual(
            self, tensor, torch.tensor([1], dtype=torch.uint8))
        TestCase.assertEqual(self, mask, torch.tensor([True]))

        self.assertRaisesRegex(
            RuntimeError,
            "Requested mask dimension 2 is bigger than dimension 1 of given NestedTensor.",
            lambda: a.to_tensor_mask(mask_dim=2))

    # TODO once .to_list() bug fixed
    @unittest.skip("Currently only supporting nested dim 1.")
    def test_multi_scalar(self):
        # TODO: add test cases
        a = nt.nested_tensor([
               torch.tensor(1),
               torch.tensor(2),
               torch.tensor(3)
           ])
        tensor, mask = a.to_tensor_mask()

        TestCase.assertEqual(self, tensor, torch.tensor([[1, 2, 3]]))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        TestCase.assertEqual(self, tensor, torch.tensor([[1, 2, 3]]))
        TestCase.assertEqual(self, mask, torch.tensor([True]))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        TestCase.assertEqual(self, tensor, torch.tensor([[1, 2, 3]]))
        TestCase.assertEqual(self, mask, torch.tensor([[True, True, True]]))

        self.assertRaisesRegex(
            RuntimeError,
            "Requested mask dimension 3 is bigger than dimension 2 of given NestedTensor.",
            lambda: a.to_tensor_mask(mask_dim=3))

    def test_single_tensor(self):
        a = nt.nested_tensor([
            torch.tensor([1])
        ])
        tensor, mask = a.to_tensor_mask()
        TestCase.assertEqual(self, tensor, torch.tensor([[1]]))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        TestCase.assertEqual(self, tensor, torch.tensor([[1]]))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        TestCase.assertEqual(self, tensor, torch.tensor([[1]]))
        TestCase.assertEqual(self, mask, torch.tensor([True]))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        TestCase.assertEqual(self, tensor, torch.tensor([[1]]))
        TestCase.assertEqual(self, mask, torch.tensor([[True]]))

        self.assertRaisesRegex(
            RuntimeError,
            "Requested mask dimension 3 is bigger than dimension 2 of given NestedTensor.",
            lambda: a.to_tensor_mask(mask_dim=3))

    def test_multi_tensor(self):
        a = nt.nested_tensor([
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([3])
        ])
        tensor, mask = a.to_tensor_mask()
        TestCase.assertEqual(self, tensor, torch.tensor([[1],
                                                         [2],
                                                         [3]]))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        TestCase.assertEqual(self, tensor, torch.tensor([[1],
                                                         [2],
                                                         [3]]))
        TestCase.assertEqual(self, mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        TestCase.assertEqual(self, tensor, torch.tensor([[1],
                                                         [2],
                                                         [3]]))
        TestCase.assertEqual(self, mask, torch.tensor([True, True, True]))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        TestCase.assertEqual(self, tensor, torch.tensor([[1],
                                                         [2],
                                                         [3]]))
        TestCase.assertEqual(
            self, mask, torch.tensor([[True], [True], [True]]))

    @torch.inference_mode()
    def test_mask_dim_too_small_error(self):
        a = nt.nested_tensor([
            torch.tensor([1, 2, ]),
            torch.tensor([3, 4, 5, 6]),
        ])

        self.assertRaisesRegex(
            RuntimeError, "Mask dimension is too small to represent data tensor.", lambda: a.to_tensor_mask(mask_dim=1))
    #
    # Group of tests to test nested_tensor_from_tensor_mask()
    #
    def test_ntftm_nested_dim_0_error(self):
        tensor = torch.tensor([])
        self.assertRaisesRegex(RuntimeError, "Nested dimension can't be 0.",
                               lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=0))

    def test_ntftm_none_passed(self):
        self.assertRaises(
            RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(None, None))
        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(
            torch.tensor([]), None))

    @torch.inference_mode()
    def test_ntftm_empty(self):
        tensor = torch.tensor([])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))
        TestCase.assertEqual(self, res_nt.nested_dim(), 1)

        res_nt = nt.nested_tensor_from_tensor_mask(
            tensor, tensor, nested_dim=1)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))
        TestCase.assertEqual(self, res_nt.nested_dim(), 1)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(
            tensor, tensor, nested_dim=2))

    def test_ntftm_empty2(self):
        tensor = torch.tensor([[], []])

        expected_nt1 = nt.nested_tensor([
            torch.tensor([]),
            torch.tensor([]),
        ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor)
        TestCase.assertEqual(self, res_nt, expected_nt1)

        res_nt = nt.nested_tensor_from_tensor_mask(
            tensor, tensor, nested_dim=1)
        TestCase.assertEqual(self, res_nt, expected_nt1)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor)
        TestCase.assertEqual(self, res_nt, expected_nt1)

        res_nt = nt.nested_tensor_from_tensor_mask(
            tensor, tensor, nested_dim=1)
        TestCase.assertEqual(self, res_nt, expected_nt1)

    def test_ntftm_empty3(self):
        tensor = torch.tensor([0])
        mask = torch.tensor(False)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))

        tensor = torch.tensor([[0], [0]])
        mask = torch.tensor([[False], [False]])

    def test_ntftm_empty_error(self):
        tensor = torch.tensor([])
        mask = torch.tensor([True])
        self.assertRaisesRegex(RuntimeError,
                               "Data tensor can't be emtpy if a mask has values.",
                               lambda: nt.nested_tensor_from_tensor_mask(tensor, mask))

        tensor = torch.tensor([1])
        mask = torch.tensor([])
        self.assertRaisesRegex(RuntimeError,
                               "Mask tensor can't be emtpy if a data tensor has values.",
                               lambda: nt.nested_tensor_from_tensor_mask(tensor, mask))

    def test_ntftm_single_scalar_mask_false(self):
        scalar = torch.tensor([1], dtype=torch.uint8)
        mask = torch.tensor(False)

        res_nt = nt.nested_tensor_from_tensor_mask(scalar, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))

    def test_ntftm_single_scalar_error(self):
        tensor = torch.tensor(1)
        mask = torch.tensor(True)
        self.assertRaisesRegex(RuntimeError, "Can't construct nested tensor from a scalar.",
                               lambda: nt.nested_tensor_from_tensor_mask(tensor, mask))

    def test_ntftm_single_scalar(self):
        tensor = torch.tensor([1], dtype=torch.float)
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([torch.tensor(1)]))

        mask = torch.tensor([True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([torch.tensor(1)]))

        # Extra dim
        tensor = torch.tensor([[1]], dtype=torch.float)
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.tensor([1])
                             ]))

    def test_ntftm_multi_scalars(self):
        tensor = torch.tensor([1, 2, 3])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.tensor(1),
                                 torch.tensor(2),
                                 torch.tensor(3)
                             ], dtype=torch.int64))

        mask = torch.tensor([True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.tensor(1),
                                 torch.tensor(2),
                                 torch.tensor(3)
                             ], dtype=torch.int64))

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(
            tensor, mask, nested_dim=2))

        # Extra dim
        tensor = torch.tensor([[1, 2, 3]])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.tensor([1, 2, 3])
                             ], dtype=torch.int64))

    def test_ntftm_single_tensor_all_true_mask(self):
        tensor = torch.tensor([[1]], dtype=torch.float)
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(
            self, res_nt, nt.nested_tensor([torch.tensor([1])]))

        mask = torch.tensor([True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(
            self, res_nt, nt.nested_tensor([torch.tensor([1])]))

    def test_ntftm_multi_tensor_scalar_true_mask(self):
        tensor = torch.tensor([[1], [2], [3]])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.tensor([1]),
                                 torch.tensor([2]),
                                 torch.tensor([3])
                             ], dtype=tensor.dtype))

        # Extra dim
        tensor = torch.tensor([[[1]], [[2]], [[3]]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        expected_res1 = nt.nested_tensor([
            torch.tensor([[1]]),
            torch.tensor([[2]]),
            torch.tensor([[3]])
        ], dtype=tensor.dtype)
        TestCase.assertEqual(self, res_nt, expected_res1)

    def test_ntftm_multi_tensor_true_mask(self):
        extected_nt1 = nt.nested_tensor([
            torch.tensor([[1]]),
            torch.tensor([[2]]),
            torch.tensor([[3]])
        ])

        tensor = torch.tensor([[[1]],
                               [[2]],
                               [[3]]], dtype=torch.float)

        # Mask dim 3
        mask3 = torch.tensor([[[True]],
                              [[True]],
                              [[True]]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask3)
        TestCase.assertEqual(self, extected_nt1, res_nt)

        # Mask dim 2
        mask2 = torch.tensor([[True],
                              [True],
                              [True]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask2)
        TestCase.assertEqual(self, extected_nt1, res_nt)

        # Mask dim 1
        mask1 = torch.tensor([True, True, True])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask1)
        TestCase.assertEqual(self, extected_nt1, res_nt)

        # Mask dim 0
        mask0 = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask0)
        TestCase.assertEqual(self, extected_nt1, res_nt)

    def test_ntftm_single_tensor_all_false_mask(self):
        tensor = torch.tensor([[1]])
        mask = torch.tensor([False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))

        tensor = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))

    def test_ntftm_multi_tensor_all_false_mask(self):
        tensor = torch.tensor([[[1], [2], [3]]])
        mask = torch.tensor([False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))

        mask = torch.tensor([False, False, False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt, nt.nested_tensor([]))

        mask = torch.tensor([[False], [False], [False]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.tensor([], dtype=tensor.dtype)
                             ], dtype=torch.int64))

    def test_ntftm_multi_tensor_all_false_mask2(self):
        tensor = torch.tensor([[[1], [2], [3]]])
        mask = torch.tensor([[[False], [False], [False]]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, res_nt,
                             nt.nested_tensor([
                                 torch.empty((3, 0), dtype=tensor.dtype)
                             ], dtype=tensor.dtype))

    def test_ntgtm_multi_scalar_mix_mask(self):
        tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float)
        mask = torch.tensor([True, False, False, True])
        expected_nt = nt.nested_tensor([
            torch.tensor(1),
            torch.tensor(4)
        ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, expected_nt, res_nt)

    def test_ntgtm_multi_tensor_mix_mask(self):
        tensor = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
        mask = torch.tensor([True, False, False, True])
        expected_nt = nt.nested_tensor([
            torch.tensor([1]),
            torch.tensor([4])
        ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, expected_nt, res_nt)

    def test_ntgtm_scalar_with_empty_mix_mask(self):
        tensor = torch.tensor([[0], [11]], dtype=torch.float)
        mask = torch.tensor([False,  True])

        expected_nt1 = nt.nested_tensor([
            torch.tensor([11], dtype=torch.long)
        ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        TestCase.assertEqual(self, expected_nt1, res_nt)

    def test_ntftm_test_multi_tensor_mix_mask(self):
        expected_nt1 = nt.nested_tensor([
            torch.tensor([1, 2, 3]),
            torch.tensor([4])
        ])

        tensor = torch.tensor([[1, 2, 3],
                               [4, 0, 0]], dtype=torch.float)
        mask = torch.tensor([[True,  True,  True],
                             [True, False, False]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        TestCase.assertEqual(self, expected_nt1, res_nt)

    def test_ntftm_test_multi_tensor_mix_mask2(self):
        expected_nt1 = nt.nested_tensor([
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4]])
        ])

        tensor = torch.tensor([[[1, 2, 3]],
                               [[4, 0, 0]]], dtype=torch.float)
        mask = torch.tensor([[[True,  True,  True]],
                             [[True, False, False]]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        TestCase.assertEqual(self, expected_nt1, res_nt)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(
            tensor, mask, nested_dim=4))

    def test_to_padded_tensor(self):
        data1 = torch.tensor(
            [[[0.8413, 0.7325, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000]],

             [[0.6334, 0.5473, 0.3273, 0.0564],
              [0.3023, 0.6826, 0.3519, 0.1804],
              [0.8431, 0.1645, 0.1821, 0.9185]]])
        mask1 = torch.tensor(
            [[[True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]],

             [[True,  True,  True,  True],
              [True,  True,  True,  True],
              [True,  True,  True,  True]]])
        nt2 = nt.nested_tensor_from_tensor_mask(data1, mask1)
        data2, mask2 = nt2.to_tensor_mask()
        self.assertEqual(data1, data2)
        self.assertEqual(mask1, mask2)
        data3 = nt2.to_padded_tensor(padding=-10)
        data1 = data1 + ~mask1 * -10
        self.assertEqual(data1, data3)


if __name__ == "__main__":
    unittest.main()
