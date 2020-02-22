__version__ = '0.0.1.dev20202222+253dd6c'
git_version = '253dd6c81d21a6b87905f6c6d81dda836b2c2d84'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
