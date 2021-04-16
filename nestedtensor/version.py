__version__ = '0.0.1+c28bf7a'
git_version = 'c28bf7a8017bbeff6e29189db46dd77575926639'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
