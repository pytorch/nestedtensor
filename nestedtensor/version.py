__version__ = '0.0.1.dev20202186+6a311e4'
git_version = '6a311e495bad9055a5240d15f2552e06a2907f18'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
