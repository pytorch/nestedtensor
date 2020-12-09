__version__ = '0.0.1+de193f7'
git_version = 'de193f7ba770000291e0e00bf0affdd7fd1d4353'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
