__version__ = '0.0.1.dev201912920+c8e6935'
git_version = 'c8e6935a62f024e8c4d67c807a82b4fac3bd7413'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
