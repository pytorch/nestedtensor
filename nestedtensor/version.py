__version__ = '0.0.1.dev2020645+a8b1d2c'
git_version = 'a8b1d2c537aaeae9df93dbe7fae5fe77d89b88f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
