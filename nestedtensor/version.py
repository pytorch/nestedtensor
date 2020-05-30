__version__ = '0.0.1.dev202053023+cd9e75a'
git_version = 'cd9e75ace51b84b16d93fa95404ad461b936c6a0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
