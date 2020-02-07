import torch
import numbers
from functools import wraps
from . import masking
from . import monkey_patch
import collections
import os

from . import utils
from . import creation

import nestedtensor

# Set this flag to true, if you want to enable additional verifications.
DEBUG = int(os.getenv("DEBUG", 1))


# -------------------------NestedTensor core---------------------------
class NestedTensor(object):
    # The attributes must match across all constiuents
    #
    # The NestedTensor's attributes then become that of its
    # constiuents.
    #
    # data must be a list of Tensors or NestedTensors
    #
    # Attributes:
    #     dim()
    #     layout
    #     device
    #     dtype
    #     requires_grad
    #     is_pinned()
    # Neighbors may share data, maybe all share data.
    # Levels of contiguity
    def __init__(self, impl):
        self._impl = impl

    # --- impl forward ---

    def dim(self):
        """
        Returns the number of dimensions of ```self``` NestedTensor.
        The dimension is defined as the dimension of the Tensor constiuents
        and the level of nesting.

        """
        return self._impl.dim()

    def is_pinned(self):
        """
        Returns true if the NestedTensor resides in pinned memory.
        """
        return self._impl.is_pinned()

    @property
    def dtype(self):
        """
        The data type of ```self``` NestedTensor.
        """
        return self._impl.dtype

    @property
    def layout(self):
        """
        The layout of ```self``` NestedTensor.
        """
        return self._impl.layout

    @property
    def device(self):
        """
        The device of ```self``` NestedTensor.
        """
        return self._impl.device

    @property
    def requires_grad(self):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return self._impl.requires_grad

    @property
    def grad(self):
        """
        This attribute is None by default and becomes a NestedTensor the
        first time a call to backward() computes gradients for self.
        The attribute will then contain the gradients computed and future
        calls to backward() will accumulate (add) gradients into it.
        """
        return NestedTensor(self._impl.grad)

    def requires_grad_(self, requires_grad=True):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return NestedTensor(self._impl.requires_grad_(requires_grad))

    def detach(self, gradient=None, retain_graph=None, create_graph=False):
        return NestedTensor(self._impl.detach(gradient, retain_graph, create_graph))

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        if gradient is None or isinstance(self._impl, gradient._impl):
            self._impl.backward(
                gradient._impl, retain_graph._impl, create_graph)
        else:
            # TODO: Test mixed case explicitly
            for t, g in zip(self.unbind(), gradient.unbind()):
                t.backward(g, retain_graph, create_graph)

    def nested_dim(self):
        """
        The nested dimension of ```self``` NestedTensor.
        The nested dimension is defined as the level of indexing required
        to reach a Tensor constiuent.
        """
        return self._impl.nested_dim()

    def tensor_dim(self):
        """
        The tensor dimension of ```self``` NestedTensor.
        The tensor dimension is defined as the dimension of the Tensor constiuents.
        """
        return self.dim() - self.nested_dim()

    def __len__(self):
        """
        The number of entries in the list ```self``` represents.
        """
        return self._impl.__len__()

    def element_size(self):
        """
        Returns the size in bytes of an individual element.
        """
        return self._impl.element_size()

    def is_contiguous(self):
        return self._impl.is_contiguous()

    def contiguous(self):
        # TODO: Test autograd support
        return NestedTensor(self._impl.contiguous())

    def size(self, dim=None):
        if dim is not None:
            return self.size()[dim]
        return tuple(self._impl.size())

    def to(self, *args, **kwargs):
        # TODO: to is currently not supported by impls due to argparsing.
        new_tensors = [t.to(*args, **kwargs) for t in self.unbind()]
        # TODO: Make contiguous by default? Heavy operation...
        # NOTE: Needs grad support, which nestedtensor.nested_tensor
        # constructor doesn't have.
        return NestedTensor(nestedtensor.as_nested_tensor(new_tensors))

    def numel(self):
        return self._impl.numel()

    def pin_memory(self):
        return NestedTensor(self._impl.pin_memory())

    def __str__(self):
        return self._impl.__str__()

    # --- impl forward ends ---

    # --- dependent on impl ---

    # TODO: More tests
    def unbind(self, dim=0):
        """
        unbind returns a tuple containing the entries
        of the list ```self``` represents. 

        For now unbind does not accept a dim argument akin
        to torch.Tensor.unbind
        """

        dim = utils._wrap_dim(self, dim)
        if dim == 0:
            return tuple(t if torch.is_tensor(t) else NestedTensor(t) for t in self._impl.unbind())
        else:
            unbound = tuple(t.unbind(dim - 1) for t in self.unbind(dim - 1))
            return tuple(creation.nested_tensor(t) for t in zip(*unbound))

    def to_tensor(self, dim=0):
        """
        Not necessarily a view.
        """
        dim = utils._wrap_dim(self, dim)
        # Convert entire NestedTensor into Tensor
        if dim == 0:
            if None in self.size():
                raise ValueError("Shape not Tensor compliant")
            return self._impl.to_tensor()
        # If dim is bigger than nested_dim the NestedTensor is already
        # of Tensor for dimensions bigger than the given.
        if self.nested_dim() == 1:
            return self
        unbound = [t.to_tensor(dim=dim - 1) for t in self.unbind()]
        return creation.nested_tensor(unbound)

    def __repr__(self):
        # TODO: This relies on the fact that repr is not implemented compliant with
        # the purpose of repr for torch.Tensor. Therefore returning str is ok.
        return self.__str__()

    def nested_size(self, dim=None):
        return self._impl.nested_size(dim)

    def nested_stride(self, dim=None):
        return self._impl.nested_stride(dim)

    # --- dependent on impl ends ---

    def __torch_function__(self, func, args=(), kwargs=None):
        _local_func = None
        if kwargs is None:
            kwargs = {}
        if func in NestedTensor.__jit_function_dispatch:
            _jit_local_func = NestedTensor.__jit_function_dispatch[func]
            impl_args = [a._impl if isinstance(
                a, NestedTensor) else a for a in args]
            impl_kwargs = {k: v._impl if isinstance(
                v, NestedTensor) else v for (k, v) in kwargs.items()}
            return NestedTensor(_jit_local_func(*impl_args, **impl_kwargs))
        if func in NestedTensor.__function_dispatch:
            _local_func = NestedTensor.__function_dispatch[func]
            return _local_func(*args, **kwargs)
        raise NotImplementedError(
            "NestedTensor doesn't support function {}".format(func))

    def __bool__(self):
        raise NotImplementedError(
            "This has not been covered by NestedTensor 0.0.1")

    def __getitem__(self, key):
        # TODO: Not covered by 0.0.2 or 0.0.1!
        # NOTE: Returns a view
        # TODO: Advanced indexing
        # TODO: Tensor-wise select
        # TODO: More testing
        if isinstance(key, numbers.Number):
            return self.unbind()[key]
        if isinstance(key, slice):
            return creation.as_nested_tensor(self.unbind()[key])
        assert isinstance(key, tuple)
        if key[0] == Ellipsis:
            raise NotImplementedError(
                "Ellipsis is not yet supported for nested dimensions")
        assert len(key) > 0
        selected_tensors = self.unbind()[key[0]]
        if len(key) == 1:
            return selected_tensors
        return creation.as_nested_tensor([t[key[1:]] for t in selected_tensors])

    def __iter__(self):
        return iter(self.unbind())

    def to_nested_tensor(self, dim=0):
        # TODO: Better errors when conversion fails.
        """
        Not a view.
        """
        if dim < self.nested_dim:
            raise ValueError("Given dimension is already nested")
        else:
            if self.nested_dim == dim:
                return creation.nested_tensor(list(t.unbind() for t in self.unbind()))
            else:
                return creation.nested_tensor(list(t.to_nested_tensor(dim - 1) for t in self.unbind()))

    def to_list(self):
        return self._impl.to_list()

    def to_tuple(self):
        return self._impl.to_tuple()

    def to_tensor_mask(self, mask_dim=None):
        """Returns a named tuple TensorMask with two tensors (tensor, mask)
        of dim equal to self.dim(). Tensor will contain all data of NestedTensor,
        expect that each tensor constiuent has been padded with 0s to equal the
        largest Tensor.

        The mask is a bool tensor with a 1-to-1 correspondence to each
        element of tensor. If an entry is True, the corresponding element
        stores data that is represented by self, if it is False it is a padding
        element. These two tensors can be used to contruct a NestedTensor, however,
        nested_dim will be lost in this process."""

        return masking.make_tensor_mask(self.to_list(), mask_dim)

    def to_padded_tensor(self, mask_dim=None, padding=-1):
        tensor, mask = masking.make_tensor_mask(self.to_list(), mask_dim)
        return tensor.masked_fill(~mask, padding)
