import torch
# if getattr(torch, "__IS_MONKEY_PATCHED_BY_NESTED_TENSOR", None) is None:
#     from .nested.monkey_patch import monkey_patch
#     torch = monkey_patch(torch)
# # Confirm that this function was only applied once
# assert torch.__IS_MONKEY_PATCHED_BY_NESTED_TENSOR == 1

from .nested.utils import tensorwise
