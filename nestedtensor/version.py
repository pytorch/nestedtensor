__version__ = '0.1.4+f91b5a2'
git_version = 'f91b5a2dd1814a3199e6ecc69993f63d58a38c1b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
