__version__ = '0.0.1.dev20207153+4c1153e'
git_version = '4c1153eb104d0f2012bb2db79c2cd0f8d64ce10c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
