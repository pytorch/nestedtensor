__version__ = '0.0.1.dev2020112423+3834f40'
git_version = '3834f400197db6c4b87274b16ca136a5187b870a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
