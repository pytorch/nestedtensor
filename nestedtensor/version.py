__version__ = '0.1.4+5fd8216'
git_version = '5fd82163f5fa71bda5dbfb7643f0f630153b5cd1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
