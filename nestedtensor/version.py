__version__ = '0.0.1.dev202071521+a39c0fe'
git_version = 'a39c0fe855f3f7e89689dca98a7aa5938eda0796'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
