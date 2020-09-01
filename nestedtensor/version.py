__version__ = '0.0.1.dev2020914+d9e04a7'
git_version = 'd9e04a7436ed2f2481611d29df8cb09d9227d3e0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
