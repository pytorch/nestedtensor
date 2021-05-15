__version__ = '0.1.4+875274d'
git_version = '875274d86be761674086752c4f746e4417ce5f21'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
