__version__ = '0.0.1.dev2019122616+6d579fb'
git_version = '6d579fbaae3ed0e16475ed868c8b89647cfdf461'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
