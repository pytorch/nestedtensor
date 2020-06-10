__version__ = '0.0.1.dev202061015+c1f551b'
git_version = 'c1f551bf05ad1bb7896433eefee8f8863ae4de99'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
