__version__ = '0.0.1+dca3305'
git_version = 'dca33052cb1fb0b52d644f616e2644ad6070467d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
