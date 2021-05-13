__version__ = '0.1.4+9d305a6'
git_version = '9d305a62ca67a08626246b83b9b2af053ea411cd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
