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
    from . import functions
    import torch

    module.jit_apply_function = torch.nested_tensor.jit_apply_function
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

    # > Python data model methods. We skip functions that torch.Tensor doesn't define.
    # --- Python binary arithmetic operations
    for function_name in codegen.get_python_binary_arithmetic_operations():
        if function_name in ['divmod']:
            continue
        set_nt_method("__" + function_name + '__', utils.pointwise())

    for function_name in codegen.get_python_binary_arithmetic_operations():
        if function_name in ['matmul', 'floordiv', 'mod', 'divmod']:
            continue
        set_nt_method("__i" + function_name + '__', utils.pointwise())

    for function_name in codegen.get_python_binary_arithmetic_operations():
        if function_name in ['matmul', 'mod', 'divmod', 'lshift', 'rshift', 'and', 'xor', 'or']:
            continue
        set_nt_method("__r" + function_name + '__', utils.pointwise())

    # --- Python unary arithmetic operations
    for function_name in ['neg', 'pos', 'abs', 'invert']:
        if function_name in ['pos']:
            continue
        set_nt_method("__" + function_name + '__', utils.pointwise())

    # --- Python rich comparison operations
    for function_name in codegen.get_python_rich_comparison_functions():
        set_nt_method("__" + function_name + '__', utils.pointwise())
    # <

    # > PyTorch pointwise operations
    # --- Module and Tensor pointwise functions and methods
    for function_name in codegen.get_pointwise_functions():
        if function_name in ['fill']:
            continue
        set_nt_method(function_name, utils.pointwise())

    for function_name in codegen.get_pointwise_functions():
        if function_name in ['fill']:
            continue
        module = set_module(module, function_name, utils.pointwise())

    # --- Tensor pointwise in-place methods
    for function_name in codegen.get_pointwise_functions():
        set_nt_method(function_name + '_', utils.pointwise())
    # <

    # > PyTorch reduction operations
    # --- Module and Tensor reductions
    for function_name in codegen.get_complete_reductions():
        module = set_module_and_nt_method(
            module, function_name, utils.reduction())

    for function_name in codegen.get_tensorwise_reductions():
        module = set_module_and_nt_method(
            module, function_name, utils.reduction(support_nested_dim=False))
    # <

    # > PyTorch conversion methods
    for function_name in codegen.get_conversion_functions():
        set_nt_method(function_name, utils.pointwise())
    # <

    # > PyTorch BLAS and LAPACK operations
    for function_name in codegen.get_blas_lapack_ops():
        if function_name in ['chain_matmul', 'lu_unpack', 'matrix_rank', 'trapz']:
            continue
        set_nt_method(function_name, utils.tensorwise())

    for function_name in codegen.get_blas_lapack_ops():
        if function_name in ['bmm', 'chain_matmul', 'cholesky', 'cholesky_inverse',
                             'cholesky_solve', 'dot', 'eig', 'geqrf', 'ger', 'inverse',
                             'det', 'logdet', 'slogdet', 'lstsq', 'lu', 'lu_solve',
                             'lu_unpack', 'matmul', 'matrix_power', 'matrix_rank',
                             'mm', 'mv', 'orgqr', 'ormqr', 'pinverse', 'qr', 'solve',
                             'svd', 'symeig', 'trapz', 'triangular_solve']:
            continue
        set_nt_method(function_name + '_', utils.tensorwise())

    for function_name in codegen.get_blas_lapack_ops():
        module = set_module(
            module, function_name, utils.tensorwise())
    # <

    # > PyTorch BLAS and LAPACK operations
    for function_name in codegen.get_other_ops():
        # Custom implementation
        if function_name in ['flatten']:
            continue
        if function_name in ['broadcast_tensors', 'cartesian_prod', 'cdist', 'combinations',
                             'einsum', 'meshgrid', 'tensordot', 'tril_indices', 'triu_indices']:
            continue
        set_nt_method(function_name, utils.tensorwise())

    for function_name in codegen.get_other_ops():
        if function_name in ['bincount', 'broadcast_tensors', 'cartesian_prod', 'cdist', 'combinations',
                             'cross', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'einsum', 'flatten', 'flip',
                             'meshgrid', 'rot90', 'tensordot', 'tril_indices', 'triu_indices', 'histc',
                             'repeat_interleave', 'roll', 'trace']:
            continue
        set_nt_method(function_name + '_', utils.tensorwise())

    for function_name in codegen.get_other_ops():
        if function_name in ['flatten']:
            module = set_module(
                module, function_name, utils.tensorwise(dim_args=[1, 2, 'start_dim', 'end_dim']))
        else:
            module = set_module(
                module, function_name, utils.tensorwise())
    # <

    # # > PyTorch random sampling operations
    for function_name in codegen.get_random_sampling_operations():
        if function_name in ['cauchy', 'exponential', 'geometric', 'log_normal', 'uniform']:
            continue
        module = set_module(
            module, function_name, utils.tensorwise())

    for function_name in codegen.get_random_sampling_operations():
        if function_name in ['cauchy', 'exponential', 'geometric', 'log_normal',
                             'normal', 'random', 'uniform']:
            continue
        set_nt_method(function_name, utils.tensorwise())

    for function_name in codegen.get_random_sampling_operations():
        set_nt_method(function_name + '_', utils.tensorwise())
    # <

    # --- WORK IN PROGRESS ---

    # TODO: low-pri: improved error reporting for signal_dim
    # > PyTorch spectral operations
    for function_name in codegen.get_fft_ops():
        module = set_module_and_nt_method(
            module, function_name, utils.tensorwise(dim_args=[1, 'signal_dim']))
    for function_name in codegen.get_stft_ops():
        module = set_module_and_nt_method(
            module, function_name, utils.tensorwise())
    # <

    # --- AD HOC
    # TODO: Write mechanisms that always check if something is about to be overwritten

    # NOTE: These are methods only.
    # TODO: detach and to should be handwritten
    for function_name in ['clone', 'detach']:
        set_nt_method(function_name, utils.tensorwise())

    # By default everything is tensorwise, but for improved semantics
    # we extend the e.g. conv2d that's broadcast to also accept images
    # without a leading batch.
    # TODO: Need to split out dim functions
    # TODO: Rerun pipelines
    for function_name in codegen.get_functionals():
        # NOTE: They have Custom implementations
        if function_name in ['conv2d', 'embedding_bag', 'linear',
                             'batch_norm', 'max_pool2d', 'interpolate']:
            continue
        if function_name in ['relu', 'relu_']:
            set_module(module.nn.functional, function_name, utils.pointwise())
        else:
            set_module(module.nn.functional, function_name, utils.tensorwise())

    set_nt_method('log_softmax', utils.tensorwise(dim_args=[1, 'dim']))

    # TODO: Might need dispatch wrapper?
    module.mv = utils.tensorwise()(torch.mv)
    module.mm = utils.tensorwise()(torch.mm)

    # --- custom functions
    # set_nt_method('mm', functions.mm)
    # set_nt_method('addmm', utils.dispatch(orig_fn=torch.Tensor.addmm)(methods.addmm))
    # setattr(module, 'addmm', utils.dispatch(orig_fn=torch.addmm)(methods.addmm))

    # TODO: This is broken
    # for function_name in ['squeeze', 'unsqueeze']:
    #     setattr(module, function_name, getattr(functions, function_name))
    #     set_nt_method(function_name, getattr(functions, function_name))

    module.nn.functional.conv2d = functions.conv2d
    module.nn.functional.max_pool2d = functions.max_pool2d
    module.nn.functional.embedding_bag = functions.embedding_bag
    module.nn.functional.linear = functions.linear
    module.nn.functional.batch_norm = functions.batch_norm
    module.nn.functional.interpolate = functions.interpolate
    # module.nn.functional.nll_loss = functions.nll_loss

    # --- custom modules
    module.nn.modules.rnn.LSTM.forward = functions.lstm_forward

    module.NestedTensor = NestedTensor

    return module
