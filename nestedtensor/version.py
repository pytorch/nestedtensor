__version__ = '0.1.4+6161ad1'
git_version = '6161ad182bea33bba4d0b0400ccb3745605b6822'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
