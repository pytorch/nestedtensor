__version__ = '0.0.1.dev202092317+10227b6'
git_version = '10227b60d2159b4b144f2436c1d187bc62fa81fc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
