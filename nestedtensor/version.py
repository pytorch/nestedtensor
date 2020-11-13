__version__ = '0.0.1.dev202011132+f8a3524'
git_version = 'f8a3524c941c0c392e071fde38d34bd859ab8b66'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
