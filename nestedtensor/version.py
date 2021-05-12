__version__ = '0.1.4+5b44084'
git_version = '5b44084a534c28a8ee7bad7e2aee2f75e1cde6ea'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
