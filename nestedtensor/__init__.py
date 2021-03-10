import torch

from .nested.creation import as_nested_tensor
from .nested.creation import nested_tensor

from .nested.masking import nested_tensor_from_tensor_mask
from .nested.masking import nested_tensor_from_padded_tensor

from .nested.nested import NestedTensor

from . import nested

USE_C_EXTENSION=False
_C = False

from . import nn

# TODO: https://github.com/pytorch/pytorch/issues/34294
# torch.cat does not call __torch_function__ properly
# from .nested.nested import _new_torch_stack as stack
# from .nested.nested import _new_torch_cat as cat
