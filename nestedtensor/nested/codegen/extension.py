import torch
from functools import wraps
from collections import namedtuple

# The function classifications (unary, binary, comparison)
# are useful to generate generic code based on certaion assumptions
# such as arity. For example, someone might implement a single function
# to efficiently implement a pointwise unary function such as cos
# and then generalize it using the list of unary functions.


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
        'fill',
        'fmod',  # Requires extra kwargs
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
        'pow',
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
        "batch_norm",
        "bilinear",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "celu",
        "cosine_embedding_loss",
        "cross_entropy",
        "ctc_loss",
        "dropout",
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
        "interpolate",
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
        "max_pool2d",
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
        "relu",
        "relu_",
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
