__version__ = '0.0.1.dev2020102723+863ee95'
git_version = '863ee9581a48966485342a276644e5d9acf6e483'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
