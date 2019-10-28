import torch
from functools import wraps
from collections import namedtuple

# The function classifications (unary, binary, comparison)
# are useful to generate generic code based on certaion assumptions
# such as arity. For example, someone might implement a single function
# to efficiently implement a pointwise unary function such as cos
# and then generalize it using the list of unary functions.


def get_python_rich_comparison_functions():
    return [
        "lt",
        "le",
        "eq",
        "ne",
        "gt",
        "ge",
    ]


def get_complete_reductions():
    funcs = [
        'all',
        'any',
        'mean',
        'prod',
        'sum',
    ]
    return funcs
