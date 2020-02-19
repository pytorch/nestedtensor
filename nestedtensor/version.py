__version__ = '0.0.1.dev20202195+1333625'
git_version = '133362572df09a0b0ae15b51bf4674df8b0fef4f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
