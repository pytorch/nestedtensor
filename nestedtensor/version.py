__version__ = '0.1.4+17d1904'
git_version = '17d1904590ad9fe488cedd1a71383e12e8a1cf33'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
