__version__ = '0.0.1.dev20206173+7e71717'
git_version = '7e71717e11c8699ee5cf5c1c08451013620833f7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
