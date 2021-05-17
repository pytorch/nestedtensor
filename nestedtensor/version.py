__version__ = '0.1.4+652bc44'
git_version = '652bc4480c80fb693f33a3c2dbf5d883abdc65b6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
