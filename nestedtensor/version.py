__version__ = '0.0.1.dev20205194+afa933d'
git_version = 'afa933d830927cd08d752d958641ca4f6d27d94e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
