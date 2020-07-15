__version__ = '0.0.1.dev20207153+a8ee6ec'
git_version = 'a8ee6ec3d3cc8b8444e0f747d764dec04ad42ca1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
