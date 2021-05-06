__version__ = '0.1.4+9b5665a'
git_version = '9b5665a387565aa1fbfc018762c6a01e0830e1f1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
