__version__ = '0.0.1.dev2020111820+5d57aef'
git_version = '5d57aefed2f89ea85c6393d9a80e3083d5097a44'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
