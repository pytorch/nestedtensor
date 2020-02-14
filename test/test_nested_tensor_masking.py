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

    def test_empty_nt(self):
        a = nt.nested_tensor([])
        tensor, mask = a.to_tensor_mask()

        self.assertEqual(mask, torch.tensor([]))
        self.assertEqual(tensor, torch.tensor([]))

        a = nt.nested_tensor([
            nt.nested_tensor([])
        ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(mask, torch.tensor([[]]))
        self.assertEqual(tensor, torch.tensor([[]]))

        a = nt.nested_tensor([
            nt.nested_tensor([]),
            nt.nested_tensor([])
        ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(mask, torch.tensor([[], []]))
        self.assertEqual(tensor, torch.tensor([[], []]))

    #TODO once bug fixed
    def test_empty_tensor(self):
        #a = nt.nested_tensor([
        #        torch.tensor([])
        #    ])
        #self.assertRaisesRegex(RuntimeError, "Empty tensors are not yet supported.", lambda: a.to_tensor_mask())

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([])
                ])
            ])
        self.assertRaisesRegex(RuntimeError, "Empty tensors are not yet supported.", lambda: a.to_tensor_mask())

    def test_single_scalar(self):
        a = nt.nested_tensor([
                torch.tensor(1, dtype=torch.uint8)
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([1], dtype=torch.uint8))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([1], dtype=torch.uint8))
        self.assertEqual(mask, torch.tensor(True))
        
        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([1], dtype=torch.uint8))
        self.assertEqual(mask, torch.tensor(True))

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=2))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1, dtype=torch.bfloat16)
                ])
            ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor([True]))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor([[True]]))

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=3))

    #TODO once bug fixed
    def test_multi_scalar(self):
        # TODO: add test cases
        #a = nt.nested_tensor([
        #        torch.tensor(1),
        #        torch.tensor(2),
        #        torch.tensor(3)
        #    ])
        #tensor, mask = a.to_tensor_mask()

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1),
                    torch.tensor(2),
                    torch.tensor(3)
                ])
            ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1, 2, 3]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1, 2, 3]]))
        self.assertEqual(mask, torch.tensor([[True, True, True]]))

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=3))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1)
                ]),
                nt.nested_tensor([
                    torch.tensor(2)
                ]),
                nt.nested_tensor([
                    torch.tensor(3)
                ])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1], [2], [3]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1], [2], [3]]))
        self.assertEqual(mask, torch.tensor([[True], [True], [True]]))

    def test_single_tensor(self):
        a = nt.nested_tensor([
                torch.tensor([1])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor(True))
        
        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor([True]))
        
        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor([[True]]))

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=3))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([1])
                ])
            ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor([True]))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor([[True]]))

        tensor, mask = a.to_tensor_mask(mask_dim=3)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor([[[True]]]))

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=4))

    def test_scalar_and_empty_nt(self):
        a = nt.nested_tensor([
            nt.nested_tensor([]),
            nt.nested_tensor([
                torch.tensor(11, dtype=torch.long)
            ])
        ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[0], [11]], dtype=torch.long))
        self.assertEqual(mask, torch.tensor([False,  True]))

    def test_scalar_and_empty_nt_cuda(self):
        a = nt.nested_tensor([
            nt.nested_tensor([], dtype=torch.long, device='cuda'),
            nt.nested_tensor([
                torch.tensor(11, dtype=torch.long, device='cuda')
            ])
        ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[0], [11]], dtype=torch.long, device='cuda'))
        self.assertEqual(mask, torch.tensor([False,  True], device='cuda'))

    def test_multi_tensor(self):
        a = nt.nested_tensor([
                torch.tensor([1]),
                torch.tensor([2]),
                torch.tensor([3])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor(True))

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor([True, True, True]))

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor([[True], [True], [True]]))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1),
                    torch.tensor(2),
                    torch.tensor(3)
                ])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1, 2, 3]]))
        self.assertEqual(mask, torch.tensor(True))
        
        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1, 2, 3]]))
        self.assertEqual(mask, torch.tensor([[True, True, True]]))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1)
                ]),
                nt.nested_tensor([
                    torch.tensor(2)
                ]),
                nt.nested_tensor([
                    torch.tensor(3)
                ])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1], [2], [3]]))
        self.assertEqual(mask, torch.tensor(True))
        
        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1], [2], [3]]))
        self.assertEqual(mask, torch.tensor([[True], [True], [True]]))

    def test_multi_tensor2(self):
        a = nt.nested_tensor([
                nt.nested_tensor([
                    nt.nested_tensor([
                        torch.tensor([[1, 2, 3, 4],
                                      [5, 6, 7, 8]])
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[0, 0], [3, 4]])
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[1]])
                    ]),
                ])
            ])

        expected_t = torch.tensor([[
            [[[1, 2, 3, 4],
              [5, 6, 7, 8]]],
            [[[0, 0, 0, 0],
              [3, 4, 0, 0]]],
            [[[1, 0, 0, 0],
              [0, 0, 0, 0]]],
        ]])

        expected_m = torch.tensor([[
            [[[ True,  True,  True,  True],
              [ True,  True,  True,  True]]],
            [[[ True,  True, False, False],
              [ True,  True, False, False]]],
            [[[ True, False, False, False],
              [False, False, False, False]]]]])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(expected_t, tensor)
        self.assertEqual(expected_m, mask)

    def test_mask_dim_too_small_error(self):
        a = nt.nested_tensor([
            torch.tensor([1, 2,]),
            torch.tensor([3, 4, 5, 6]),
        ])

        self.assertRaisesRegex(RuntimeError, "Mask dimention is too small to represent data tensor.", lambda: a.to_tensor_mask(mask_dim=1))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    nt.nested_tensor([
                        torch.tensor([[1, 2, 3, 4],
                                      [5, 6, 7, 8]])
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[0, 0], [3, 4]])
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[1]])
                    ]),
                ])
            ])
        
        for dim in range(4):
            self.assertRaisesRegex(RuntimeError, "Mask dimention is too small to represent data tensor.", lambda: a.to_tensor_mask(mask_dim=dim))

'''
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

    def test_to_tensor_mask_empty_nt(self):
        original_nt = nt.nested_tensor([])
        res_tensor, res_mask = original_nt.to_tensor_mask()
        self.assertEqual(res_tensor, torch.tensor([]))
        self.assertEqual(res_mask, torch.tensor([]))

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
        print(mask)
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

    #
    # Group of tests to test to_tensor_mask() and nested_tensor_from_tensor_mask()
    # with corner cases and complex nested tensors
    #

    def test_scalar(self):
        scalar = torch.tensor(3)
        original_nt = nt.nested_tensor([scalar])

        res_tensor, res_mask = original_nt.to_tensor_mask()
        new_nt = nt.nested_tensor_from_tensor_mask(res_tensor, res_mask, nested_dim=original_nt.nested_dim())
        self.assertEqual(res_tensor, torch.tensor([3]))
        self.assertEqual(res_mask, torch.tensor([True]))
        self.assertEqual(new_nt, original_nt)

        original_nt = nt.nested_tensor([
            nt.nested_tensor([
                scalar
            ]),
            nt.nested_tensor([
                scalar,
                scalar
            ])
        ])
        res_tensor, res_mask = original_nt.to_tensor_mask()
        new_nt = nt.nested_tensor_from_tensor_mask(res_tensor, res_mask, nested_dim=original_nt.nested_dim())
        self.assertEqual(new_nt, original_nt)
        self.assertEqual(res_mask, torch.tensor([[True, False], [True, True]]))
        self.assertEqual(res_tensor, torch.tensor([[3, 0], [3, 3]]))

    def test_nested_tensor_from_tensor_mask_empty_tensor(self):
        nt1 = nt.nested_tensor([
            nt.nested_tensor([
                torch.tensor([])
            ]),
            nt.nested_tensor([
                torch.tensor([])
            ])
        ])

        self.assertRaises(RuntimeError, lambda: nt1.to_tensor_mask())

    def test_complex_nt(self):  
        nt1 = nt.nested_tensor([
            nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([1], dtype=torch.float),
                    torch.tensor([1], dtype=torch.float)
                ]),
                nt.nested_tensor([
                    torch.tensor([2, 3], dtype=torch.float)
                ]),
            ])
        ])

        expected_tensor = torch.tensor([[[[1., 0.],
                                          [1., 0.]],
                                         [[2., 3.],
                                          [0., 0.]]]])

        expected_mask = torch.tensor([[[[ True, False],
                                        [ True, False]],
                                       [[ True,  True],
                                        [False, False]]]])

        tensor, maskNone = nt1.to_tensor_mask()
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, maskNone, nested_dim=nt1.nested_dim())

        self.assertEqual(tensor, expected_tensor)
        self.assertEqual(tensor.dtype, torch.float)
        self.assertEqual(tensor.layout, torch.strided)
        self.assertEqual(maskNone, expected_mask)

        tensor, mask0 = nt1.to_tensor_mask(mask_dim=0)
        tensor, mask1 = nt1.to_tensor_mask(mask_dim=1)
        tensor, mask2 = nt1.to_tensor_mask(mask_dim=2)
        self.assertEqual(mask0, maskNone)
        self.assertEqual(mask0, mask1)
        self.assertEqual(mask0, mask2)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, maskNone, nested_dim=nt1.nested_dim())
        self.assertEqual(res_nt, nt1)
    '''

if __name__ == "__main__":
    unittest.main()
