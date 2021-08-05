__version__ = '0.1.4+da883d9'
git_version = 'da883d94a7cb250db7ec7d6d152764e6e8e8788a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
