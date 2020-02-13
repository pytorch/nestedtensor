from __future__ import absolute_import, division, print_function, unicode_literals

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
from utils import nested_size_to_list

class TestTensorMask(TestCase):
    def test_gen_nested_tensor(self):
        nt1 = utils.gen_nested_tensor(seed=1, nested_dim=1, tensor_dim=1, size_low=2, size_high=2)
        self.assertTrue(nt.is_nested_tensor(nt1))
        self.assertEqual(nt1.nested_dim(), 1)
        self.assertEqual(nt1.size(), (2, 2))
        self.assertEqual(nested_size_to_list(nt1.nested_size()), [[2],[2]])
        self.assertEqual(nt1[0].dim(), 1)
        self.assertEqual(nt1[0].size(), torch.Size([2]))

        nt1 = utils.gen_nested_tensor(seed=1, nested_dim=2, tensor_dim=1, size_low=2, size_high=2)
        self.assertTrue(nt.is_nested_tensor(nt1))
        self.assertEqual(nt1.nested_dim(), 2)
        self.assertEqual(nt1.size(), (2, 2, 2))
        self.assertEqual(nt1[0].dim(), 2)
        self.assertEqual(nested_size_to_list(nt1.nested_size()), [[[2],[2]], [[2],[2]]])

        nt1 = utils.gen_nested_tensor(seed=3, nested_dim=2, tensor_dim=2, size_low=1, size_high=5)
        self.assertTrue(nt.is_nested_tensor(nt1))
        self.assertEqual(nt1.nested_dim(), 2)
        self.assertEqual(nt1.size(), (3, 2, None, None))
        self.assertEqual(nt1[0].dim(), 3)
        self.assertEqual(nt1[0].size(), (2, None, None))
        self.assertEqual(nt1[0][0].size(), torch.Size([2, 4]))
        self.assertEqual(nt1[0][0][0].size(), torch.Size([4]))
        self.assertEqual(nested_size_to_list(nt1.nested_size()), [[[2, 4], [4, 3]], [[2, 4], [4, 3]], [[2, 4], [4, 3]]])

    #
    # Group of tests to test to_tensor_mask() 
    # 
    def test_to_tensor_mask_nt_dim_2(self):
        nt1 = nt.nested_tensor([
            torch.tensor([1, 2]),
            torch.tensor([3, 4])
        ])
        exp_tensor = torch.tensor([[1, 2], [3, 4]])
        exp_mask = torch.tensor(True)

        res_tensor, res_mask = nt1.to_tensor_mask()
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

    def test_to_tensor_mask_nt_dim_4(self):
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

        # mask_dim = None
        res_tensor, res_mask = original_nt2.to_tensor_mask()
        exp_mask = torch.tensor([[[ True, False]],
                                 [[ True, True]]])
        exp_tensor = torch.tensor([[[[1, 2], [0, 0]]],
                                   [[[3, 4], [5, 6]]]])

        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 0
        res_tensor, res_mask = original_nt2.to_tensor_mask(mask_dim=0)
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 1
        res_tensor, res_mask = original_nt2.to_tensor_mask(mask_dim=1)
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 2
        res_tensor, res_mask = original_nt2.to_tensor_mask(mask_dim=2)
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 3
        res_tensor, res_mask = original_nt2.to_tensor_mask(mask_dim=3)
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 4
        res_tensor, res_mask = original_nt2.to_tensor_mask(mask_dim=4)
        exp_tensor = torch.tensor([[[[1, 2], [0, 0]]],
                                    [[[3, 4], [5, 6]]]])
        exp_mask = torch.tensor([[[[ True,  True], [False, False]]],
                                 [[[ True,  True], [ True,  True]]]])
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

    def test_to_tensor_mask_nt_dim_5(self):
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

        # mask_dim = None
        res_tensor, res_mask = original_nt.to_tensor_mask()
        exp_tensor = torch.tensor([[[[[1, 2], [3, 4]]]],
                                   [[[[5, 6], [7, 8]]]]])
        exp_mask = torch.tensor(True)
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 0
        res_tensor, res_mask = original_nt.to_tensor_mask(mask_dim=0)
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 1
        res_tensor, res_mask = original_nt.to_tensor_mask(mask_dim=1)
        exp_tensor = torch.tensor([[[[[1, 2], [3, 4]]]],
                                   [[[[5, 6], [7, 8]]]]])
        exp_mask = torch.tensor([True, True])
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)

        # mask_dim = 2
        res_tensor, res_mask = original_nt.to_tensor_mask(mask_dim=2)
        exp_mask = torch.tensor([[True], [True]])
        exp_tensor = torch.tensor([[[[[1, 2], [3, 4]]]],
                                   [[[[5, 6], [7, 8]]]]])
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)
        
        # mask_dim = 3
        res_tensor, res_mask = original_nt.to_tensor_mask(mask_dim=3)
        exp_mask = torch.tensor([[[True]], [[True]]])
        exp_tensor = torch.tensor([[[[[1, 2], [3, 4]]]],
                                   [[[[5, 6], [7, 8]]]]])
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor)    

        # mask_dim = 4
        res_tensor, res_mask = original_nt.to_tensor_mask(mask_dim=4)
        exp_mask = torch.tensor([[[[True, True]]],[[[True, True]]]])
        exp_tensor = torch.tensor([[[[[1, 2], [3, 4]]]],
                                   [[[[5, 6], [7, 8]]]]])
        self.assertEqual(res_mask, exp_mask)
        self.assertEqual(res_tensor, exp_tensor) 

    #
    # Group of tests to test nested_tensor_from_tensor_mask() 
    #
    def test_nested_tensor_from_tensor_mask_dim_none(self):
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

        tensor, mask = original_nt.to_tensor_mask()
        self.assertRaisesRegex(RuntimeError, 
                               "Mask has to have dimention > 0", 
                               lambda: nt.nested_tensor_from_tensor_mask(tensor, mask))

    def test_nested_tensor_from_tensor_mask_dim_1(self):
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

        tensor, mask = original_nt.to_tensor_mask(mask_dim = 1)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=0)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)

        # nested_dim is bigger than masks dimension 
        self.assertRaises(ValueError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2))
        self.assertRaises(ValueError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3))

    def test_nested_tensor_from_tensor_mask_dim_2(self):
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

        tensor, mask = original_nt.to_tensor_mask(mask_dim=2)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=0)
        self.assertEqual(expected_nt0, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)

        # nested_dim is bigger than masks dimension 
        self.assertRaises(ValueError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3))

    def test_nested_tensor_from_tensor_mask_dim_3(self):
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
