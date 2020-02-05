__version__ = '0.0.1.dev2020254+e0ee3e0'
git_version = 'e0ee3e08aa2dec6a927754a6263b37305cac94ae'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
