__version__ = '0.0.1.dev201912719+b404e0f'
git_version = 'b404e0faf7a0a976fbf76d215b60079086a674c6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
