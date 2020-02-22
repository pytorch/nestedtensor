__version__ = '0.0.1.dev20202221+66b1d46'
git_version = '66b1d46275f28a03b6468933228b58d5bc511b6a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
