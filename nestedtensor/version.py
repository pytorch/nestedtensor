__version__ = '0.1.4+e727321'
git_version = 'e7273219a5b68fb5f4831537f6b67c3cac0bdcd6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
