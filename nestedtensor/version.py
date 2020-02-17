__version__ = '0.0.1.dev20202173+3049361'
git_version = '3049361e1e235a0c3d8ba5a4067bcef0ab8914dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
