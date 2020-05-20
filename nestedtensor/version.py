__version__ = '0.0.1.dev202052020+0f7ff52'
git_version = '0f7ff52b180474623b39e84631571a4d046fce98'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
