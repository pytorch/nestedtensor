import torch
# if getattr(torch, "__IS_MONKEY_PATCHED_BY_NESTED_TENSOR", None) is None:
#     from .nested.monkey_patch import monkey_patch
#     torch = monkey_patch(torch)
# # Confirm that this function was only applied once
# assert torch.__IS_MONKEY_PATCHED_BY_NESTED_TENSOR == 1

from .nested.utils import tensorwise
from .nested.utils import is_nested_tensor

from .nested.creation import as_nested_tensor
from .nested.creation import nested_tensor

from .nested.masking import nested_tensor_from_tensor_mask
from .nested.masking import nested_tensor_from_padded_tensor

from .nested.nested import NestedTensor

from .nested.monkey_patch import monkey_patch


torch.tensorwise = tensorwise
torch.is_nested_tensor = is_nested_tensor

# > PyTorch constructors
torch.as_nested_tensor = as_nested_tensor
torch.nested_tensor = nested_tensor
torch.nested_tensor_from_tensor_mask = nested_tensor_from_tensor_mask
torch.nested_tensor_from_padded_tensor = nested_tensor_from_padded_tensor

torch.NestedTensor = NestedTensor

monkey_patch(torch.NestedTensor)
