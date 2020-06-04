__version__ = '0.0.1.dev2020643+3509160'
git_version = '3509160a5ac7fb786bb1a7eb5e026072d5d04bf0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
