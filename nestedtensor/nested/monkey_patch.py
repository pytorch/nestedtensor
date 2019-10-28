def _check_meaningful_overwrite(cls, method_name):
    import os
    DEBUG = int(os.getenv("DEBUG", 0))

    class DefaultClass(object):
        pass

    if DEBUG and getattr(cls, method_name, False) and not getattr(DefaultClass, method_name, False):
        raise Exception("WARNING: " + method_name + " already exists "
                        "and not part of default class")


def set_module(module, name, wrapper):
    new_fn = wrapper(getattr(module, name))
    new_fn.__MODULE_FUNCTION_SUPPORTED_BY_NESTED_TENSOR = 1
    setattr(module, name, new_fn)
    return module


def set_nt_method(name, wrapper):
    import torch
    from .nested import NestedTensor
    _check_meaningful_overwrite(NestedTensor, name)
    setattr(NestedTensor, name, wrapper(getattr(torch.Tensor, name)))


def set_module_and_nt_method(module, name, wrapper):
    set_nt_method(name, wrapper)
    return set_module(module, name, wrapper)


def monkey_patch(module):
    """
    Functions that are being skipped are sometimes not skipped
    for a good reason other than a lack of completed implemetation
    of the torch.Tensor or torch module corresponding implementations.
    """

    import os
    DEBUG = int(os.getenv("DEBUG", 0))
    from . import utils
    from .nested import NestedTensor
    from . import creation
    from . import masking
    from . import codegen
    import torch

    module.__IS_MONKEY_PATCHED_BY_NESTED_TENSOR = getattr(
        module, "__IS_MONKEY_PATCHED_BY_NESTED_TENSOR", 0) + 1
    module.tensorwise = utils.tensorwise
    module.is_nested_tensor = utils.is_nested_tensor

    # > PyTorch constructors
    module.as_nested_tensor = creation.as_nested_tensor
    module.nested_tensor = creation.nested_tensor
    module.nested_tensor_from_tensor_mask = masking.nested_tensor_from_tensor_mask
    module.nested_tensor_from_padded_tensor = masking.nested_tensor_from_padded_tensor
    # <

    # > PyTorch reduction operations
    # --- Module and Tensor reductions
    for function_name in codegen.get_complete_reductions():
        module = set_module_and_nt_method(
            module, function_name, utils.reduction())
    # <

    # --- Python rich comparison operations
    for function_name in codegen.get_python_rich_comparison_functions():
        set_nt_method("__" + function_name + '__', utils.pointwise())
    # <

    module.NestedTensor = NestedTensor

    return module
