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
    #
    # Group of tests to test to_tensor_mask() 
    # 
    '''
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
    # Group of tests to test nested_tensor_from_tensor_mask() 
    #
    def test_ntftm_nested_dim_0_error(self):
        tensor = torch.tensor([])
        self.assertRaisesRegex(RuntimeError, "Nested dimention can't be 0.", lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=0))

    def test_ntftm_empty(self):
        tensor = torch.tensor([])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor)
        self.assertEqual(res_nt, nt.nested_tensor([]))

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=1)
        self.assertEqual(res_nt, nt.nested_tensor([]))

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=2))

        tensor = torch.tensor([[], []])
        
        expected_nt1 = nt.nested_tensor([
            torch.tensor([]),
            torch.tensor([]),
        ])

        expected_nt2 = nt.nested_tensor([
            nt.nested_tensor([]),
            nt.nested_tensor([])
        ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor)
        self.assertEqual(res_nt, expected_nt1)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=1)
        self.assertEqual(res_nt, expected_nt1)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=2)
        self.assertEqual(res_nt, expected_nt2)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=3))

    def test_scalar_mask_false(self):
        scalar = torch.tensor([1], dtype=torch.uint8)
        mask = torch.tensor(False)
        self.assertRaisesRegex(RuntimeError, "Scalar mask cant be False.", lambda: nt.nested_tensor_from_tensor_mask(scalar, mask))
    
    def atest_foo(self):
        a = nt.nested_tensor([
            nt.nested_tensor([]),
            nt.nested_tensor([])
        ])

        tensor, mask = a.to_tensor_mask()
        #print(a.nested_dim()) == 2

        a = nt.nested_tensor([
            torch.tensor([]),
            torch.tensor([])
        ])
        tensor, mask = a.to_tensor_mask()
        #print(a.nested_dim()) == 1

        scalar = torch.tensor([1], dtype=torch.uint8)
        print(scalar.dim())
        print(scalar.numel())
        print('\n')
        
        scalar = torch.tensor(1, dtype=torch.uint8)
        print(scalar.dim())
        print(scalar.numel())
        print('\n')
        
        scalar = torch.tensor([], dtype=torch.uint8)
        print(scalar.dim())
        print(scalar.numel())

        mask = torch.tensor(True)
        res_nt =  nt.nested_tensor_from_tensor_mask(scalar, mask)
        print(res_nt)

if __name__ == "__main__":
    unittest.main()
