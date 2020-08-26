__version__ = '0.0.1.dev202082621+d554e58'
git_version = 'd554e583a10525297642ae27805635c08bbca85e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
