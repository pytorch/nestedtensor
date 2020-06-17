__version__ = '0.0.1.dev20206174+07a02ee'
git_version = '07a02ee7d9d738d0a0b8579ef219f3202797bedb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
