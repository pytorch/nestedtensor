__version__ = '0.1.4+6d72a29'
git_version = '6d72a2980342eab716a908c1c0f751c30631d8a9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
