__version__ = '0.0.1.dev20205131+ef4fe9f'
git_version = 'ef4fe9f9e5bf9a21837227b5d29ea8543e102352'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
