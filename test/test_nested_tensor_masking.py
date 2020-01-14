import traceback
import functools
import pdb
import sys
import torch
import nestedtensor as NT
import unittest
from utils import TestCase
import random

import utils

class TestTensorMask(TestCase):
    def test_gen_nested_tensor(self):
        nt = utils.gen_nested_tensor(seed=1, nested_dim=1, tensor_dim=1, size_low=2, size_high=2)
        self.assertTrue(NT.is_nested_tensor(nt))
        self.assertEqual(nt.nested_dim(), 1)
        self.assertEqual(nt.size(), (2, 2))
        self.assertEqual(nt[0].dim(), 1)
        self.assertEqual(nt[0].size(), torch.Size([2]))

        nt = utils.gen_nested_tensor(seed=1, nested_dim=2, tensor_dim=1, size_low=2, size_high=2)
        self.assertTrue(NT.is_nested_tensor(nt))
        self.assertEqual(nt.nested_dim(), 2)
        self.assertEqual(nt.size(), (2, 2, 2))
        self.assertEqual(nt[0].dim(), 1)
        self.assertEqual(nt[0].size(), (2, 2))
        self.assertEqual(nt[0][0].size(), torch.Size([2]))

        nt = utils.gen_nested_tensor(seed=1, nested_dim=2, tensor_dim=2, size_low=2, size_high=2)
        self.assertTrue(NT.is_nested_tensor(nt))
        self.assertEqual(nt.nested_dim(), 2)
        self.assertEqual(nt.size(), (2, 2, 2, 2))
        self.assertEqual(nt[0].dim(), 1)
        self.assertEqual(nt[0].size(), (2, 2, 2))
        self.assertEqual(nt[0][0].size(), torch.Size([2, 2]))
        self.assertEqual(nt[0][0][0].size(), torch.Size([2]))

        nt = utils.gen_nested_tensor(seed=1, nested_dim=2, tensor_dim=2, size_low=10, size_high=10)
        self.assertTrue(NT.is_nested_tensor(nt))
        self.assertEqual(nt.nested_dim(), 2)
        self.assertEqual(nt.size(), (10, 10, 10, 10))
        self.assertEqual(nt[0].dim(), 1)
        self.assertEqual(nt[0].size(), (10, 10, 10))
        self.assertEqual(nt[0][0].size(), torch.Size([10, 10]))
        self.assertEqual(nt[0][0][0].size(), torch.Size([10]))

    def test_to_tensor_mask(self):
        nt = utils.gen_nested_tensor(1, 1, 1, size_low=2, size_high=2)
        tensor, mask = nt.to_tensor_mask()
        self.assertEqual(mask, torch.tensor(True))
        self.assertEqual(mask.size(), torch.Size([]))
        self.assertEqual(mask.dim(), 0)
        self.assertEqual(tensor.size(), torch.Size([2, 2]))
        self.assertEqual(tensor.dim(), 2)

        nt = NT.nested_tensor([
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([1, 2])
                ]),
            ]),
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([3, 4]),
                    torch.tensor([5, 6])
                ])
            ])
        ])
        
        # mask_dim = None
        tensor, mask = nt.to_tensor_mask()
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True,  True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        # mask_dim = 0
        tensor, mask = nt.to_tensor_mask(mask_dim=0)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True,  True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        # mask_dim = 1
        tensor, mask = nt.to_tensor_mask(mask_dim=1)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True,  True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        # mask_dim = 2
        tensor, mask = nt.to_tensor_mask(mask_dim=2)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True,  True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        # mask_dim = 3
        tensor, mask = nt.to_tensor_mask(mask_dim=3)
        self.assertEqual(mask, torch.tensor([[[ True, False]],
                                             [[ True,  True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        # mask_dim = 4
        tensor, mask = nt.to_tensor_mask(mask_dim=4)
        self.assertEqual(mask, torch.tensor([[[[ True,  True],
                                               [False, False]]],
                                             [[[ True,  True],
                                               [ True,  True]]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(mask.dim(), 4)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 4)

        nt = NT.nested_tensor([
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([[1, 2], 
                                  [3, 4]])
                ]),
            ]),
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([[5, 6],
                                  [7, 8]])
                ]),
            ])
        ])

        # mask_dim = None
        tensor, mask = nt.to_tensor_mask()
        self.assertEqual(mask, torch.tensor(True))
        self.assertEqual(mask.size(), torch.Size([]))
        self.assertEqual(mask.dim(), 0)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)

        # mask_dim = 0
        tensor, mask = nt.to_tensor_mask(mask_dim=0)
        self.assertEqual(mask, torch.tensor(True))
        self.assertEqual(mask.size(), torch.Size([]))
        self.assertEqual(mask.dim(), 0)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)

        # mask_dim = 1
        tensor, mask = nt.to_tensor_mask(mask_dim=1)
        self.assertEqual(mask, torch.tensor([True, True]))
        self.assertEqual(mask.size(), torch.Size([2]))
        self.assertEqual(mask.dim(), 1)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)

        # mask_dim = 2
        tensor, mask = nt.to_tensor_mask(mask_dim=2)
        self.assertEqual(mask, torch.tensor([[True], [True]]))
        self.assertEqual(mask.size(), torch.Size([2, 1]))
        self.assertEqual(mask.dim(), 2)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)

        # mask_dim = 3
        tensor, mask = nt.to_tensor_mask(mask_dim=3)
        self.assertEqual(mask, torch.tensor([[[True]], [[True]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 1]))
        self.assertEqual(mask.dim(), 3)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)

        # mask_dim = 4
        tensor, mask = nt.to_tensor_mask(mask_dim=4)
        self.assertEqual(mask, torch.tensor([[[[True, True]]], [[[True, True]]]]))
        self.assertEqual(mask.size(), torch.Size([2, 1, 1, 2]))
        self.assertEqual(mask.dim(), 4)
        self.assertEqual(tensor.size(), torch.Size([2, 1, 1, 2, 2]))
        self.assertEqual(tensor.dim(), 5)

    def test_nested_tensor_from_tensor_mask(self):
        original_nt = NT.nested_tensor([
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([[1, 2], 
                                  [3, 4]])
                ]),
            ]),
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([[5, 6],
                                  [7, 8]])
                ]),
            ])
        ])

        # 
        # mask_dim = 4
        #

        # nested_dim = 4
        tensor, mask = original_nt.to_tensor_mask(mask_dim=4)
        nt4 = NT.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=4)
        self.assertEqual(original_nt, nt4)

        # nested_dim = 3
        nt3 = NT.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        expected_nt3 = NT.nested_tensor([
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([[1, 2],
                                  [3, 4]]),
                ]),
            ]),
            NT.nested_tensor([
                NT.nested_tensor([
                    torch.tensor([[5, 6],
                                 [7, 8]]),
                ]),
            ]),
        ])
        self.assertEqual(expected_nt3, nt3)

        # nested_dim = 2
        nt2 = NT.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        expected_nt2 = NT.nested_tensor([
            NT.nested_tensor([
                torch.tensor([[[1, 2],
                               [3, 4]]]),
            ]),
            NT.nested_tensor([
                torch.tensor([[[5, 6],
                              [7, 8]]]),
            ]),
        ])
        self.assertEqual(expected_nt2, nt2)

        # nested_dim = 1
        nt1 = NT.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        expected_nt1 = NT.nested_tensor([
            torch.tensor([[[[1, 2],
                            [3, 4]]]]),
            torch.tensor([[[[5, 6],
                            [7, 8]]]]),
        ])
        self.assertEqual(expected_nt1, nt1)

        # nested_dim = 0
        nt0 = NT.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=0)
        expected_nt0 = torch.tensor([
                                        [[[[1, 2],
                                        [3, 4]]]],
                                        [[[[5, 6],
                                        [7, 8]]]]
                                    ])
        self.assertEqual(expected_nt0, nt0)

        # nested_dim = None
        nt = NT.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt0, nt)

if __name__ == "__main__":
    unittest.main()


# TODO: 
# 1. test cases for all asserts
# 2. test cases for all errors 
# 3. test cases for NT from tensor+mask
# 4. ask if tensor(True) mask + tensor -> NT? its an error now.
#
#