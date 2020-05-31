__version__ = '0.0.1.dev202053116+ad13a81'
git_version = 'ad13a81c511a5f00ea58a1ef30bb97f21d614bf0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
