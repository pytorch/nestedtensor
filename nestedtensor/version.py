__version__ = '0.1.4+3fd184e'
git_version = '3fd184ed10d331d2160d8ea4876a26f4b5cf9a9f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
