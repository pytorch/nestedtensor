__version__ = '0.1.4+bfa7a20'
git_version = 'bfa7a208c3c591fc4b4bbf168eae8c5093206499'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
