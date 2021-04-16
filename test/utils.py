import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
import random
import urllib

from utils_test_case import TestCase


def debug_on(*exceptions):
    if not exceptions:
        exceptions = (BaseException,)

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


def internet_on():
    try:
        urllib.request.urlopen("http://www.google.com", timeout=1)
        return True
    except urllib.error.URLError as err:
        return False


def _shape_prod(shape_):
    shape = tuple(shape_)
    start = 1
    for s in shape:
        start = start * s
    return start


def random_float_tensor(seed, size, a=22695477, c=1, m=2 ** 32, requires_grad=False):
    """ Generates random tensors given a seed and size
    https://en.wikipedia.org/wiki/Linear_congruential_generator
    X_{n + 1} = (a * X_n + c) % m
    Using Borland C/C++ values
     The tensor will have values between [0,1)
    Inputs:
        seed (int): an int
        size (Tuple[int]): the size of the output tensor
        a (int): the multiplier constant to the generator
        c (int): the additive constant to the generator
        m (int): the modulus constant to the generator
    """
    num_elements = 1
    for s in size:
        num_elements *= s

    arr = [(a * seed + c) % m]
    for i in range(num_elements - 1):
        arr.append((a * arr[i] + c) % m)

    return torch.tensor(arr, requires_grad=requires_grad).float().view(size) / m


def random_int_tensor(seed, size, low=0, high=2 ** 32, a=22695477, c=1, m=2 ** 32):
    """ Same as random_float_tensor but integers between [low, high)
    """
    return (
        torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low
    ).to(torch.int64)


def gen_float_tensor(seed, shape, requires_grad=False):
    return random_float_tensor(seed, shape, requires_grad=requires_grad)


def gen_random_int(seed, low=0, high=2 ** 32):
    """ Returns random integer in [low, high)
    """
    return int(random_int_tensor(seed, (), low=low, high=high))


# TODO: Something occasionally causes a NaN here...
def gen_nested_list(seed, nested_dim, tensor_dim, size_low=1, size_high=10):
    tensors = []
    num_tensors = gen_random_int(
        (seed * nested_dim + seed) * 1024, low=size_low, high=size_high
    )
    assert nested_dim > 0
    if nested_dim == 1:
        for i in range(num_tensors):
            ran = gen_random_int(
                (seed * nested_dim + seed) * (1024 * i), low=size_low, high=size_high
            )
            ran_size = ()
            for _ in range(tensor_dim):
                ran = gen_random_int(ran * 1024, low=size_low, high=size_high)
                ran_size = ran_size + (ran,)

            tensors.append(gen_float_tensor(ran, ran_size))
    else:
        for _ in range(num_tensors):
            tensors.append(
                gen_nested_list(
                    num_tensors * seed,
                    nested_dim - 1,
                    tensor_dim,
                    size_low=size_low,
                    size_high=size_high,
                )
            )
    return tensors


def nested_map(fn, data):
    if isinstance(data, list):
        return [nested_map(fn, d) for d in data]
    else:
        return fn(data)


def gen_nested_tensor(
    seed, nested_dim, tensor_dim, size_low=1, size_high=10, constructor=None
):
    if constructor is None:
        constructor = nestedtensor.as_nested_tensor
    return constructor(
        gen_nested_list(
            seed, nested_dim, tensor_dim, size_low=size_low, size_high=size_high
        )
    )


def get_first_tensor(nested_list):
    if isinstance(nested_list, list):
        return get_first_tensor(nested_list[0])
    else:
        return nested_list

def get_nn_C_functions():
    return [
        "relu",
        "relu_",
        "dropout",
        "conv2d",
        "max_pool2d",
        "batch_norm",
        "cross_entropy",
        "interpolate",
    ]

def get_unary_C_functions():
    return [
        "abs",
        "acos",
        "angle",
        "asin",
        "atan",
        "bitwise_not",
        "ceil",
        "conj",
        "cos",
        "cosh",
        "digamma",
        "erf",
        "erfc",
        "erfinv",
        "exp",
        "expm1",
        "floor",
        "frac",
        "imag",
        "inverse",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_not",
        "neg",
        "nonzero",
        "real",
        "reciprocal",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
    ]


def get_unary_functions():
    return [
        'abs',
        'acos',
        'asin',
        'atan',
        'ceil',
        'clamp',  # Requires extra kwargs
        'clamp_min',  # Undocumented
        'clamp_max',  # Undocumented
        'cos',
        'cosh',
        'digamma',
        'erf',
        'erfc',
        'erfinv',
        'exp',
        'expm1',
        'floor',
        # 'fill', Not a unary op
        # 'fmod',  # Requires extra kwargs
        'frac',
        # 'hardshrink', # TODO: Not part of aten
        'lgamma',
        'log',
        'log10',
        'log1p',
        'log2',
        'mvlgamma',
        'neg',
        # 'nonzero', # TODO: Special case because it modifies dtype - no inplace
        # 'polygamma', # TODO: Undocumented and first argument not Tensor
        #  polygamma NOTE: Should change to dispatch on first tensor argument not argument - but then raises questions of mixed tensor vs. nestedtensor etc.
        # 'prelu', # TODO: no prelu_out in aten
        'reciprocal',
        # 'relu', # TODO: no relu_out in aten
        # 'renorm', # TODO: Requires extra kwargs
        'round',
        'rsqrt',
        'sigmoid',
        'sign',
        'sin',
        'sinh',
        'sqrt',
        'tan',
        'tanh',
        'trunc']


def get_binary_functions():
    return [
        'add',
        'mul',
        'sub',
        'div',
        # 'pow',
        'atan2',
        'remainder',
    ]


def get_python_rich_comparison_functions():
    return [
        "lt",
        "le",
        "eq",
        "ne",
        "gt",
        "ge",
    ]


def get_pointwise_functions():
    funcs = []
    funcs += get_unary_functions()
    funcs += get_binary_functions()
    funcs += get_python_rich_comparison_functions()
    return funcs


def get_python_binary_arithmetic_operations():
    funcs = [
        "add",
        "sub",
        "mul",
        "matmul",
        "truediv",
        "floordiv",
        "mod",
        "divmod",
        "pow",
        "lshift",
        "rshift",
        "and",
        "xor",
        "or",
    ]
    return funcs


def get_complete_reductions():
    funcs = [
        'all',
        'any',
        'mean',
        'prod',
        'sum',
    ]
    return funcs


def get_random_sampling_operations():
    funcs = [
        "bernoulli",
        "cauchy",
        "exponential",
        "geometric",
        "log_normal",
        "normal",
        "random",
        "uniform",
    ]
    return funcs


def get_tensorwise_reductions():
    # Only support per-tensor or full reductions.
    funcs = [
        'argmax',
        'argmin',
        'argsort',
        'cumprod',
        'cumsum',
        'std',
        'var',
        'max',  # may return tuple
        'median',  # may return tuple
        'min',  # may return tuple
        'mode',  # returns tuple
    ]
    return funcs


def get_conversion_functions():
    # Convenience functions for to(torch.float) and such
    funcs = [
        "bfloat16",
        "bool",
        "byte",
        "char",
        "cpu",
        "cuda",
        "double",
        "float",
        "half",
        "int",
        "long",
        "short",
        "to_dense",
        "to_mkldnn",
        "to_sparse",
    ]
    return funcs


def get_fft_ops():
    funcs = [
        "fft",
        "ifft",
        "rfft",
        "irfft",
    ]
    return funcs


def get_stft_ops():
    funcs = [
        "stft",
    ]
    return funcs


def get_blas_lapack_ops():
    """
    These functions all have fixed dimension inputs,
    which makes it easy to think about for NestedTensors
    """
    funcs = [
        # BLAS and LAPACK functions
        "addbmm",
        "addmm",
        "addmv",
        "addr",
        "baddbmm",
        "bmm",
        "chain_matmul",
        "cholesky",
        "cholesky_inverse",
        "cholesky_solve",
        "dot",
        "eig",
        "geqrf",
        "ger",
        "inverse",
        "det",
        "logdet",
        "slogdet",
        "lstsq",
        "lu",
        "lu_solve",
        "lu_unpack",
        "matmul",
        "matrix_power",
        "matrix_rank",
        "mm",
        "mv",
        "orgqr",
        "ormqr",
        "pinverse",
        "qr",
        "solve",
        "svd",
        "symeig",
        "trapz",
        "triangular_solve",
    ]
    return funcs


def get_other_ops():
    """
    Misc functions based on other classification in torch docs.
    """
    funcs = [
        "bincount",
        "broadcast_tensors",
        "cartesian_prod",
        "cdist",
        "combinations",
        "cross",
        "diag",
        "diag_embed",
        "diagflat",
        "diagonal",
        "einsum",
        "flatten",
        "flip",
        "rot90",
        "histc",
        "meshgrid",
        "renorm",
        "repeat_interleave",
        "roll",
        "tensordot",
        "trace",
        "tril",
        "tril_indices",
        "triu",
        "triu_indices",
    ]
    return funcs


def get_functionals():
    funcs = [
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
        "adaptive_max_pool1d_with_indices",
        "adaptive_max_pool2d_with_indices",
        "adaptive_max_pool3d_with_indices",
        "affine_grid",
        "alpha_dropout",
        "assert_int_or_pair",
        #"batch_norm",
        "bilinear",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "celu",
        "cosine_embedding_loss",
        #"cross_entropy",
        "ctc_loss",
        #"dropout",
        "dropout2d",
        "dropout3d",
        "elu",
        "embedding",
        "embedding_bag",
        "feature_alpha_dropout",
        "fold",
        "fractional_max_pool2d_with_indices",
        "fractional_max_pool3d_with_indices",
        "gelu",
        "glu",
        "grid_sample",
        "group_norm",
        "gumbel_softmax",
        "hardshrink",
        "hardtanh",
        "hinge_embedding_loss",
        "instance_norm",
        #"interpolate",
        "kl_div",
        "l1_loss",
        "layer_norm",
        "leaky_relu",
        "linear",
        "local_response_norm",
        "log_softmax",
        "lp_pool1d",
        "lp_pool2d",
        "max_pool1d",
        #"max_pool2d",
        "max_pool3d",
        "margin_ranking_loss",
        "max_pool1d_with_indices",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
        "max_unpool1d",
        "max_unpool2d",
        "max_unpool3d",
        "mse_loss",
        "multi_head_attention_forward",
        "multilabel_margin_loss",
        "multilabel_soft_margin_loss",
        "multi_margin_loss",
        "nll_loss",
        "normalize",
        "pad",
        "pairwise_distance",
        "poisson_nll_loss",
        "prelu",
        #"relu",
        #"relu_",
        "relu6",
        "rrelu",
        "selu",
        "sigmoid",
        "smooth_l1_loss",
        "soft_margin_loss",
        "softmax",
        "softmin",
        "softsign",
        "tanh",
        "tanhshrink",
        "threshold",
        "triplet_margin_loss",
        "unfold",
        "upsample",
        "upsample_bilinear",
        "upsample_nearest",
    ]
    return funcs
