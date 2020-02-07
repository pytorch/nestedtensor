__version__ = '0.0.1.dev2020276+020175b'
git_version = '020175b841546047fece68dd792578aed24231be'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
