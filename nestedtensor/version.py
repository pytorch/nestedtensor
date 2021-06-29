__version__ = '0.1.4+9c48e2b'
git_version = '9c48e2b26c7d7a878efa0fdbe1756277c9e19746'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
