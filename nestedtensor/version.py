__version__ = '0.0.1.dev202061217+ed567bf'
git_version = 'ed567bfe8937579a91f0d145f125b5c1cad3c487'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
