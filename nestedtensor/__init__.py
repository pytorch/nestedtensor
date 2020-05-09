import torch

# from .nested.utils import tensorwise
from .nested.utils import is_nested_tensor

from .nested.creation import as_nested_tensor
from .nested.creation import nested_tensor

from .nested.masking import nested_tensor_from_tensor_mask
from .nested.masking import nested_tensor_from_padded_tensor

from .nested.nested import NestedTensor

# from .nested.monkey_patch import monkey_patch

from . import nested

from . import _C

# nested.monkey_patch.monkey_patch(NestedTensor)
