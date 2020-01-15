import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
import random

from utils_test_case import TestCase



def debug_on(*exceptions):
    if not exceptions:
        exceptions = (BaseException, )

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


def _shape_prod(shape_):
    shape = tuple(shape_)
    start = 1
    for s in shape:
        start = start * s
    return start


def random_float_tensor(seed, size, a=22695477, c=1, m=2 ** 32,
                        requires_grad=False):
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
    return (torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low).to(torch.int64)


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
        (seed * nested_dim + seed) * 1024, low=size_low, high=size_high)
    assert nested_dim > 0
    if nested_dim == 1:
        for i in range(num_tensors):
            ran = gen_random_int((seed * nested_dim + seed)
                                 * (1024 * i), low=size_low, high=size_high)
            ran_size = ()
            for _ in range(tensor_dim):
                ran = gen_random_int(ran * 1024, low=size_low, high=size_high)
                ran_size = ran_size + (ran,)

            tensors.append(gen_float_tensor(ran, ran_size))
    else:
        for i in range(num_tensors):
            tensors.append(gen_nested_list(
                num_tensors * seed, nested_dim - 1, tensor_dim))
    return tensors


def nested_map(fn, data):
    if isinstance(data, list):
        return [nested_map(fn, d) for d in data]
    else:
        return fn(data)


def gen_nested_tensor(seed, nested_dim, tensor_dim, size_low=1, size_high=10):
    return nestedtensor.nested_tensor(gen_nested_list(seed, nested_dim, tensor_dim, size_low=size_low, size_high=size_high))
