__version__ = '0.0.1+c4ef2a5'
git_version = 'c4ef2a5e1c5ad0083b33455f601dd1f6ba4614a9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
