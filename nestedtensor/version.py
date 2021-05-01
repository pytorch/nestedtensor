__version__ = '0.1.4+5f1717f'
git_version = '5f1717f46dad03e91dff6c3db39a47c9599e2d6d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
