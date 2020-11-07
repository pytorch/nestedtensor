__version__ = '0.0.1.dev20201172+750259e'
git_version = '750259ee4554affd4f0149c58aecb213c6a9544d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
