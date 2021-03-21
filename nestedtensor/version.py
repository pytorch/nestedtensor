__version__ = '0.0.1+949d29d'
git_version = '949d29d8f84c946a7e08c1d2bb52fe320de8e078'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
