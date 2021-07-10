__version__ = '0.1.4+7be8164'
git_version = '7be8164298f17dd9f510af0155ab7c4c50359d4e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
