__version__ = '0.1.4+6ca0b54'
git_version = '6ca0b549c679feeb71b887f4f418a5f9bd1e30c5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
