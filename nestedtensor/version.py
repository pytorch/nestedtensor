__version__ = '0.1.4+00af8da'
git_version = '00af8da1eceb54ba356812cbf818c701e1cc518a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
