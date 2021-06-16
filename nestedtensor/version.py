__version__ = '0.1.4+93cc2b4'
git_version = '93cc2b41bccbb8e783c2223e5b7d76f431baa39d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
