__version__ = '0.0.1.dev2019121322+6a0048b'
git_version = '6a0048bf8a5980248ce63f334934deabfd76fc97'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
