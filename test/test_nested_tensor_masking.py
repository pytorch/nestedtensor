import traceback
import functools
import pdb
import sys
import torch
import nestedtensor as nt
import unittest
import random
import utils
from utils import TestCase
from utils import nested_size_to_list

def check_tensor_options(self, expected, actual):
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(expected.device, actual.device)
    self.assertEqual(expected.layout, actual.layout)
    self.assertEqual(expected.requires_grad, actual.requires_grad)

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
    def test_empty_nt(self):
        a = nt.nested_tensor([])
        tensor, mask = a.to_tensor_mask()

        self.assertEqual(mask, torch.tensor(False))
        self.assertEqual(tensor, torch.tensor([0]))
        check_tensor_options(self, tensor, a)

        a = nt.nested_tensor([
            nt.nested_tensor([])
        ])

        tensor, mask = a.to_tensor_mask()

        self.assertEqual(mask, torch.tensor(False))
        self.assertEqual(tensor, torch.tensor([[0]]))
        check_tensor_options(self, tensor, a)

        a = nt.nested_tensor([
            nt.nested_tensor([]),
            nt.nested_tensor([])
        ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(mask, torch.tensor(False))
        self.assertEqual(tensor, torch.tensor([[0], [0]]))
        check_tensor_options(self, tensor, a)

    #TODO once .to_list() bug fixed
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
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([1], dtype=torch.uint8))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([1], dtype=torch.uint8))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=2))

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1, dtype=torch.bfloat16)
                ])
            ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor([True]))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1]], dtype=torch.bfloat16))
        self.assertEqual(mask, torch.tensor([[True]]))
        check_tensor_options(self, tensor, a)

        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=3))

    #TODO once .to_list() bug fixed
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
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1, 2, 3]]))
        self.assertEqual(mask, torch.tensor([[True, True, True]]))
        check_tensor_options(self, tensor, a)

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
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1], [2], [3]]))
        self.assertEqual(mask, torch.tensor([[True], [True], [True]]))
        check_tensor_options(self, tensor, a)

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
        check_tensor_options(self, tensor, a)

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
        check_tensor_options(self, tensor, a)

    def test_single_tensor(self):
        a = nt.nested_tensor([
                torch.tensor([1])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)
        
        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)
        
        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor([True]))
        check_tensor_options(self, tensor, a)
        
        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1]]))
        self.assertEqual(mask, torch.tensor([[True]]))
        check_tensor_options(self, tensor, a)
        
        self.assertRaisesRegex(RuntimeError, "Mask dimention is bigger than nested dimention of a nested tensor.", lambda: a.to_tensor_mask(mask_dim=3))

        # Extra dim
        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([1])
                ])
            ])

        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor([True]))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor([[True]]))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=3)
        self.assertEqual(tensor, torch.tensor([[[1]]]))
        self.assertEqual(mask, torch.tensor([[[True]]]))
        check_tensor_options(self, tensor, a)

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
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=0)
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=1)
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor([True, True, True]))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[1],
                                               [2],
                                               [3]]))
        self.assertEqual(mask, torch.tensor([[True], [True], [True]]))
        check_tensor_options(self, tensor, a)

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([1]),
                    torch.tensor([2]),
                    torch.tensor([3])
                ])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[[1], [2], [3]]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[[1], [2], [3]]]))
        self.assertEqual(mask, torch.tensor([[True, True, True]]))
        check_tensor_options(self, tensor, a)

        a = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([1])
                ]),
                nt.nested_tensor([
                    torch.tensor([2])
                ]),
                nt.nested_tensor([
                    torch.tensor([3])
                ])
            ])
        tensor, mask = a.to_tensor_mask()
        self.assertEqual(tensor, torch.tensor([[[1]], [[2]], [[3]]]))
        self.assertEqual(mask, torch.tensor(True))
        check_tensor_options(self, tensor, a)

        tensor, mask = a.to_tensor_mask(mask_dim=2)
        self.assertEqual(tensor, torch.tensor([[[1]], [[2]], [[3]]]))
        self.assertEqual(mask, torch.tensor([[True], [True], [True]]))
        check_tensor_options(self, tensor, a)

    def test_multi_tensor2(self):
        a = nt.nested_tensor([
                nt.nested_tensor([
                    nt.nested_tensor([
                        torch.tensor([[1, 2, 3, 4],
                                      [5, 6, 7, 8]], dtype=torch.bfloat16, device='cuda', requires_grad=True)
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[0, 0], [3, 4]], dtype=torch.bfloat16, device='cuda', requires_grad=True)
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[1]], dtype=torch.bfloat16, device='cuda', requires_grad=True)
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
        check_tensor_options(self, tensor, a)

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

    #
    # Group of tests to test nested_tensor_from_tensor_mask() 
    #
    def test_ntftm_nested_dim_0_error(self):
        tensor = torch.tensor([])
        self.assertRaisesRegex(RuntimeError, "Nested dimention can't be 0.", lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=0))

    def test_ntftm_none_passed(self):
        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(None, None))
        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(torch.tensor([]), None))

    def test_ntftm_empty(self):
        tensor = torch.tensor([])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        self.assertEqual(res_nt.nested_dim(), 1)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=1)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        self.assertEqual(res_nt.nested_dim(), 1)
        check_tensor_options(self, tensor, res_nt)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=2))

    def test_ntftm_empty2(self):
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
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=1)
        self.assertEqual(res_nt, expected_nt1)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=2)
        self.assertEqual(res_nt, expected_nt2)
        check_tensor_options(self, tensor, res_nt)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, tensor, nested_dim=3))

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
        self.assertRaisesRegex(RuntimeError, "Scalar mask cant be False.", lambda: nt.nested_tensor_from_tensor_mask(scalar, mask))

    def test_ntftm_single_scalar_error(self):
        tensor = torch.tensor(1)
        mask = torch.tensor(True)
        self.assertRaisesRegex(RuntimeError, "Can't construct nested tensor from a scalar.", lambda: nt.nested_tensor_from_tensor_mask(tensor, mask))

    def test_ntftm_single_scalar(self):
        tensor = torch.tensor([1])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([torch.tensor(1)]))
        check_tensor_options(self, tensor, res_nt)

        mask = torch.tensor([True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([torch.tensor(1)]))
        check_tensor_options(self, tensor, res_nt)

        # Extra dim
        tensor = torch.tensor([[1]])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            torch.tensor([1])
                        ]))
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor(1)
                            ])
                        ]))
        check_tensor_options(self, tensor, res_nt)

    def test_ntftm_multi_scalars(self):
        tensor = torch.tensor([1, 2, 3])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            torch.tensor(1),
                            torch.tensor(2),
                            torch.tensor(3)
                        ]))
        check_tensor_options(self, tensor, res_nt)

        mask = torch.tensor([True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            torch.tensor(1),
                            torch.tensor(2),
                            torch.tensor(3)
                        ]))

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2))

        # Extra dim
        tensor = torch.tensor([[1, 2, 3]])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            torch.tensor([1, 2, 3])
                        ]))
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor(1),
                                torch.tensor(2),
                                torch.tensor(3)
                            ])
                        ]))
        check_tensor_options(self, tensor, res_nt)

    def test_ntftm_single_tensor_all_true_mask(self):
        tensor = torch.tensor([[1]])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([torch.tensor([1])]))
        check_tensor_options(self, tensor, res_nt)

        mask = torch.tensor([True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([torch.tensor([1])]))
        check_tensor_options(self, tensor, res_nt)

    def test_ntftm_multi_tensor_scalar_true_mask(self):
        tensor = torch.tensor([[1], [2], [3]])
        mask = torch.tensor(True)
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                        nt.nested_tensor([
                            torch.tensor([1]),
                            torch.tensor([2]),
                            torch.tensor([3])
                        ]))
        check_tensor_options(self, tensor, res_nt)

        # Extra dim
        tensor = torch.tensor([[[1]], [[2]], [[3]]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        expected_res1 = nt.nested_tensor([
                            torch.tensor([[1]]),
                            torch.tensor([[2]]),
                            torch.tensor([[3]])
                        ])
        self.assertEqual(res_nt, expected_res1)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        expected_res2 = nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor([1])
                            ]),
                            nt.nested_tensor([
                                torch.tensor([2])
                            ]),
                            nt.nested_tensor([
                                torch.tensor([3])
                            ])
                        ])
        self.assertEqual(res_nt, expected_res2)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        expected_res3 = nt.nested_tensor([
                            nt.nested_tensor([
                                nt.nested_tensor([
                                    torch.tensor(1)
                                ])
                            ]),
                            nt.nested_tensor([
                                nt.nested_tensor([
                                    torch.tensor(2)
                                ])
                            ]),
                            nt.nested_tensor([
                                nt.nested_tensor([
                                    torch.tensor(3)
                                ])
                            ])
                        ])
        self.assertEqual(res_nt, expected_res3)
        check_tensor_options(self, tensor, res_nt)

    def test_ntftm_multi_tensor_true_mask(self):
        extected_nt1 = nt.nested_tensor([
                    torch.tensor([[1]]),
                    torch.tensor([[2]]),
                    torch.tensor([[3]])
                ])

        extected_nt2 = nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor([1])
                            ]),
                            nt.nested_tensor([
                                torch.tensor([2])
                            ]),
                            nt.nested_tensor([
                                torch.tensor([3])
                            ])
                        ])

        tensor = torch.tensor([[[1]],
                               [[2]],
                               [[3]]])

        # Mask dim 3 
        mask3 = torch.tensor([[[True]],
                              [[True]],
                              [[True]]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask3)
        self.assertEqual(extected_nt1, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask3, nested_dim=2)
        self.assertEqual(extected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)

        # Mask dim 2 
        mask2 = torch.tensor([[True],
                              [True],
                              [True]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask2)
        self.assertEqual(extected_nt1, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask2, nested_dim=2)
        self.assertEqual(extected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)

        # Mask dim 1
        mask1 = torch.tensor([True, True, True])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask1)
        self.assertEqual(extected_nt1, res_nt)
        check_tensor_options(self, tensor, res_nt)
        
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask1, nested_dim=2)
        self.assertEqual(extected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)

    def test_ntftm_single_tensor_all_false_mask(self): 
        tensor = torch.tensor([[1]])
        mask = torch.tensor([False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        #TODO: uncomment once bug with dt is fixed
        #check_tensor_options(self, tensor, res_nt)

        tensor = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        #TODO: uncomment once bug with dt is fixed
        #check_tensor_options(self, tensor, res_nt)

    def test_ntftm_multi_tensor_all_false_mask(self):
        tensor = torch.tensor([[[1], [2], [3]]])
        mask = torch.tensor([False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        #TODO: uncomment once bug with dt is fixed
        #check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        #TODO: uncomment once bug with dt is fixed
        #check_tensor_options(self, tensor, res_nt)

        mask = torch.tensor([False, False, False])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, nt.nested_tensor([]))
        #TODO: uncomment once bug with dt is fixed
        #check_tensor_options(self, tensor, res_nt)

        mask = torch.tensor([[False], [False], [False]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                         nt.nested_tensor([
                             torch.tensor([], dtype=tensor.dtype)
                         ]))
        #TODO: uncomment once bug with dt is fixed
        #check_tensor_options(self, tensor, res_nt)

    def test_ntftm_multi_tensor_all_false_mask2(self):
        tensor = torch.tensor([[[1], [2], [3]]])
        mask = torch.tensor([[[False], [False], [False]]])
        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(res_nt, 
                         nt.nested_tensor([
                            torch.empty((3, 0), dtype=tensor.dtype)
                         ]))
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(res_nt, 
                         nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor([], dtype=tensor.dtype),
                                torch.tensor([], dtype=tensor.dtype),
                                torch.tensor([], dtype=tensor.dtype)
                            ])
                         ]))
        check_tensor_options(self, tensor, res_nt)

    def test_ntgtm_multi_scalar_mix_mask(self):
        tensor = torch.tensor([1, 2, 3, 4])
        mask = torch.tensor([True, False, False, True])
        expected_nt = nt.nested_tensor([
                        torch.tensor(1),
                        torch.tensor(4)
                       ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt, res_nt)
        check_tensor_options(self, tensor, res_nt)

    def test_ntgtm_multi_tensor_mix_mask(self):
        tensor = torch.tensor([[1], [2], [3], [4]])
        mask = torch.tensor([True, False, False, True])
        expected_nt = nt.nested_tensor([
                        torch.tensor([1]),
                        torch.tensor([4])
                       ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt, res_nt)
        check_tensor_options(self, tensor, res_nt)

    def test_ntgtm_scalar_with_empty_mix_mask(self):
        tensor = torch.tensor([[0], [11]])
        mask = torch.tensor([False,  True])

        expected_nt1 = nt.nested_tensor([
            torch.tensor([11], dtype=torch.long)
        ])

        expected_nt2 = nt.nested_tensor([
            nt.nested_tensor([]),
            nt.nested_tensor([
                torch.tensor(11, dtype=torch.long)
            ])
        ])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask)
        self.assertEqual(expected_nt1, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)
        
    def test_ntftm_test_multi_tensor_mix_mask(self):
        expected_nt1 = nt.nested_tensor([
                torch.tensor([1, 2, 3]),
                torch.tensor([4])
            ])

        expected_nt2 = nt.nested_tensor([
            nt.nested_tensor([
                    torch.tensor(1),
                    torch.tensor(2),
                    torch.tensor(3)
            ]),
            nt.nested_tensor([
                    torch.tensor(4)
            ])
        ])

        tensor = torch.tensor([[1, 2, 3],
                               [4, 0, 0]])
        mask = torch.tensor([[ True,  True,  True],
                             [ True, False, False]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)

    def test_ntftm_test_multi_tensor_mix_mask2(self):
        expected_nt1 = nt.nested_tensor([
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4]])
        ])

        expected_nt2 = nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([1, 2, 3])
                ]),
                nt.nested_tensor([
                    torch.tensor([4])
                ])
            ])

        expected_nt3 = nt.nested_tensor([
            nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(1),
                    torch.tensor(2),
                    torch.tensor(3)
                ])
            ]),
            nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor(4)
                ])
            ])
        ])

        tensor = torch.tensor([[[1, 2, 3]],
                               [[4, 0, 0]]])
        mask = torch.tensor([[[ True,  True,  True]],
                             [[ True, False, False]]])

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1)
        self.assertEqual(expected_nt1, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        self.assertEqual(expected_nt3, res_nt)
        check_tensor_options(self, tensor, res_nt)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=4))

    def test_ntftm_test_multi_tensor_mix_mask3(self):
        expected_nt2 = nt.nested_tensor([
            nt.nested_tensor([
                torch.tensor([[[1, 2, 3, 4],
                               [5, 6, 7, 8]]]),
                torch.tensor([[[0, 0],
                               [3, 4]]]),
                torch.tensor([[[1]]])
            ])
        ])

        expected_nt3 = nt.nested_tensor([
                nt.nested_tensor([
                    nt.nested_tensor([
                        torch.tensor([[1, 2, 3, 4],
                                      [5, 6, 7, 8]])
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[0, 0], 
                                      [3, 4]])
                    ]),
                    nt.nested_tensor([
                        torch.tensor([[1]])
                    ]),
                ])
            ])

        expected_nt4 = nt.nested_tensor([
            nt.nested_tensor([
                nt.nested_tensor([
                    nt.nested_tensor([
                        torch.tensor([1, 2, 3, 4]),
                        torch.tensor([5, 6, 7, 8])
                    ])
                ]),
                nt.nested_tensor([
                        nt.nested_tensor([
                            torch.tensor([0, 0]),
                            torch.tensor([3, 4])
                        ])
                ]),
                nt.nested_tensor([
                        nt.nested_tensor([
                            torch.tensor([1]),
                            torch.tensor([], dtype=torch.long)
                        ])
                ])
            ])
        ])

        expected_nt5 = nt.nested_tensor([
            nt.nested_tensor([
                nt.nested_tensor([
                    nt.nested_tensor([
                        nt.nested_tensor([
                            torch.tensor(1),
                            torch.tensor(2),
                            torch.tensor(3),
                            torch.tensor(4)
                        ]),
                        nt.nested_tensor([
                            torch.tensor(5),
                            torch.tensor(6),
                            torch.tensor(7),
                            torch.tensor(8)
                        ]),
                    ])
                ]),
                nt.nested_tensor([
                        nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor(0),
                                torch.tensor(0)
                            ]),
                            nt.nested_tensor([
                                torch.tensor(3),
                                torch.tensor(4)
                            ])
                        ])
                ]),
                nt.nested_tensor([
                        nt.nested_tensor([
                            nt.nested_tensor([
                                torch.tensor(1)
                            ]),
                            nt.nested_tensor([
                            ])
                        ])
                ])
            ])
        ])

        tensor = torch.tensor([
            [
                [[[1, 2, 3, 4],
                  [5, 6, 7, 8]]],
                [[[0, 0, 0, 0],
                  [3, 4, 0, 0]]],
                [[[1, 0, 0, 0],
                  [0, 0, 0, 0]]],
            ]
        ])

        mask = torch.tensor([[
            [[[ True,  True,  True,  True],
              [ True,  True,  True,  True]]],
            [[[ True,  True, False, False],
              [ True,  True, False, False]]],
            [[[ True, False, False, False],
              [False, False, False, False]]]]])

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=1))

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=2)
        self.assertEqual(expected_nt2, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=3)
        self.assertEqual(expected_nt3, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=4)
        self.assertEqual(expected_nt4, res_nt)
        check_tensor_options(self, tensor, res_nt)

        res_nt = nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=5)
        self.assertEqual(expected_nt5, res_nt)
        check_tensor_options(self, tensor, res_nt)

        self.assertRaises(RuntimeError, lambda: nt.nested_tensor_from_tensor_mask(tensor, mask, nested_dim=6))

    def test_ntftm_mask_dim(self):
        a = nt.nested_tensor([
            nt.nested_tensor([
                nt.nested_tensor([
                    torch.tensor([[1, 2, 3, 4],
                                  [5, 6, 7, 8]], dtype=torch.float16, device='cuda', requires_grad=False)
                ]),
                nt.nested_tensor([
                    torch.tensor([[1, 2, 3, 4],
                                  [5, 6, 7, 8]], dtype=torch.float16, device='cuda', requires_grad=False)
                ]),
                nt.nested_tensor([
                    torch.tensor([[1, 2, 3, 4],
                                  [5, 6, 7, 8]], dtype=torch.float16, device='cuda', requires_grad=False)
                ]),
            ])
        ])

        for i in range(a.dim()):
            t, m = a.to_tensor_mask(mask_dim=i)
            res_nt = nt.nested_tensor_from_tensor_mask(t, m, nested_dim=a.nested_dim())
            self.assertEqual(a, res_nt)
            self.assertEqual(res_nt.nested_dim(), a.nested_dim())
            check_tensor_options(self, a, res_nt)

if __name__ == "__main__":
    unittest.main()
