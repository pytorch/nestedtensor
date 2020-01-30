__version__ = '0.0.1.dev20201301+2b7c84d'
git_version = '2b7c84d2d89638cc628dfb3432d4a1397b23389a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
