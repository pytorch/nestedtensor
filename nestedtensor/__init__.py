import torch

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
