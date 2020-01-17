import traceback
import functools
import pdb
import sys
import torch
import nestedtensor as nt
import unittest
from utils import TestCase
import random
import utils

#
# Test data
#
original_nt = nt.nested_tensor([
    nt.nested_tensor([
        nt.nested_tensor([
            torch.tensor([[1, 2], 
                           [3, 4]])
        ]),
     ]),
    nt.nested_tensor([
        nt.nested_tensor([
            torch.tensor([[5, 6],
                          [7, 8]])
        ]),
    ])
])

original_nt2 = nt.nested_tensor([
    nt.nested_tensor([
        nt.nested_tensor([
            torch.tensor([1, 2])
        ]),
    ]),
    nt.nested_tensor([
        nt.nested_tensor([
            torch.tensor([3, 4]),
            torch.tensor([5, 6])
        ])
    ])
])
        
expected_nt0 = torch.tensor([[[[[1, 2],
                                [3, 4]]]],
                             [[[[5, 6],
                                [7, 8]]]]])

expected_nt1 = nt.nested_tensor([
    torch.tensor([[[[1, 2],
                    [3, 4]]]]),
    torch.tensor([[[[5, 6],
                    [7, 8]]]]),
])

expected_nt2 = nt.nested_tensor([
    nt.nested_tensor([
        torch.tensor([[[1, 2],
                       [3, 4]]]),
    ]),
    nt.nested_tensor([
        torch.tensor([[[5, 6],
                      [7, 8]]]),
    ]),
])

class TestTensorMask(TestCase):
    def test_gen_nested_tensor(self):
        nt1 = utils.gen_nested_tensor(seed=1, nested_dim=1, tensor_dim=1, size_low=2, size_high=2)
        self.assertTrue(nt.is_nested_tensor(nt1))
        self.assertEqual(nt1.nested_dim(), 1)
        self.assertEqual(nt1.size(), (2, 2))
        self.assertEqual(nt1[0].dim(), 1)
        self.assertEqual(nt1[0].size(), torch.Size([2]))

        nt1 = utils.gen_nested_tensor(seed=1, nested_dim=2, tensor_dim=1, size_low=2, size_high=2)
        self.assertTrue(nt.is_nested_tensor(nt1))
        self.assertEqual(nt1.nested_dim(), 2)
        self.assertEqual(nt1.size(), (2, 2, 2))
        self.assertEqual(nt1[0].dim(), 2)
        self.assertEqual(nt1.nested_size(), nt.NestedSize([[[2],[2]], [[2],[2]]]))

        nt1 = utils.gen_nested_tensor(seed=1, nested_dim=2, tensor_dim=2, size_low=2, size_high=2)
        self.assertTrue(nt.is_nested_tensor(nt1))
        self.assertEqual(nt1.nested_dim(), 2)
        self.assertEqual(nt1.size(), (2, 2, 2, 2))
        self.assertEqual(nt1[0].dim(), 3)
        self.assertEqual(nt1[0].size(), (2, 2, 2))
        self.assertEqual(nt1[0][0].size(), torch.Size([2, 2]))
        self.assertEqual(nt1[0][0][0].size(), torch.Size([2]))
        self.assertEqual(nt1.nested_size(), nt.NestedSize([[[2, 2],[2, 2]], [[2, 2],[2, 2]]]))

    #
    # Group of tests to test to_tensor_mask() 
    # 
    def test_to_tensor_mask_nt_dim_2(self):
        nt1 = nt.nested_tensor([
            torch.tensor([1, 2]),
            torch.tensor([3, 4])
        ])
        tensor, mask = nt1.to_tensor_mask()
        self.assertEqual(mask, torch.tensor(True))
        self.assertEqual(mask.size(), torch.Size([]))
        self.assertEqual(mask.dim(), 0)
        self.assertEqual(tensor.size(), torch.Size([2, 2]))
        self.assertEqual(tensor.dim(), 2)
        self.assertEqual(tensor.dim(), nt1.dim())

    def test_to_tensor_mask_nt_dim_4(self):
        # mask_dim = None
        tensor, mask = original_nt2.to_tensor_mask()
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True, True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.dim(), original_nt2.dim())

        # mask_dim = 0
        tensor, mask = original_nt2.to_tensor_mask(mask_dim=0)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True, True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.dim(), original_nt2.dim())

        # mask_dim = 1
        tensor, mask = original_nt2.to_tensor_mask(mask_dim=1)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True, True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        # mask_dim = 2
        tensor, mask = original_nt2.to_tensor_mask(mask_dim=2)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True,  True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.dim(), original_nt2.dim())

        # mask_dim = 3
        tensor, mask = original_nt2.to_tensor_mask(mask_dim=3)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True, True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.dim(), original_nt2.dim())

        # mask_dim = 4
        tensor, mask = original_nt2.to_tensor_mask(mask_dim=4)
        self.assertEqual(mask, torch.tensor([[[[ True, True],
                                               [False, False]]],
                                             [[[ True, True],
                                               [ True, True]]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(mask.dim(), 4)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.dim(), original_nt2.dim())

    def test_to_tensor_mask_nt_dim_5(self):
        # mask_dim = None
        tensor, mask = original_nt.to_tensor_mask()
        self.assertEqual(mask, torch.tensor(True))
        self.assertEqual(mask.size(), torch.Size([]))
        self.assertEqual(mask.dim(), 0)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)
        self.assertEqual(tensor.dim(), original_nt.dim())

        # mask_dim = 0
        tensor, mask = original_nt.to_tensor_mask(mask_dim=0)
        self.assertEqual(mask, torch.tensor(True))
        self.assertEqual(mask.size(), torch.Size([]))
        self.assertEqual(mask.dim(), 0)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)
        self.assertEqual(tensor.dim(), original_nt.dim())

        # mask_dim = 1
        tensor, mask = original_nt.to_tensor_mask(mask_dim=1)
        self.assertEqual(mask, torch.tensor([True, True]))
        self.assertEqual(mask.size(), torch.Size([2]))
        self.assertEqual(mask.dim(), 1)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)
        self.assertEqual(tensor.dim(), original_nt.dim())

        # mask_dim = 2
        tensor, mask = original_nt.to_tensor_mask(mask_dim=2)
        self.assertEqual(mask, torch.tensor([[True], [True]]))
        self.assertEqual(mask.size(), torch.Size([2, 1]))
        self.assertEqual(mask.dim(), 2)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)
        self.assertEqual(tensor.dim(), original_nt.dim())

        # mask_dim = 3
        tensor, mask = original_nt.to_tensor_mask(mask_dim=3)
        self.assertEqual(mask, torch.tensor([[[True]], [[True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 1]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)
        self.assertEqual(tensor.dim(), original_nt.dim())

        # mask_dim = 4
        tensor, mask = original_nt.to_tensor_mask(mask_dim=4)
        self.assertEqual(mask, torch.tensor([[[[True, True]]], [[[True, True]]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 1, 2]))
        self.assertEqual(mask.dim(), 4)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)
        self.assertEqual(tensor.dim(), original_nt.dim())

    #
    # Group of tests to test nested_tensor_from_tensor_mask() 
    #
    def test_nested_tensor_from_tensor_mask_dim_none(self):
        tensor, mask = original_nt.to_tensor_mask()
        self.assertRaisesRegex(RuntimeError, 
                               "Mask has to have dimention > 0", 
                               lambda: nt.nested_tensor_from_tensor_mask(tensor, mask))

    def test_nested_tensor_from_tensor_mask_dim_1(self):
        tensor, mask = original_nt.to_tensor_mask(mask_dim = 1)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=0)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt1, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        self.assertEqual(expected_nt1, res_nt)

    def test_nested_tensor_from_tensor_mask_dim_2(self):
        tensor, mask = original_nt.to_tensor_mask(mask_dim=2)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=0)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        self.assertEqual(expected_nt2, res_nt)

    def test_nested_tensor_from_tensor_mask_dim_3(self):
        tensor, mask = original_nt.to_tensor_mask(mask_dim=3)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt0, res_nt)
        
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=0)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        self.assertEqual(original_nt, res_nt)

if __name__ == "__main__":
    unittest.main()
