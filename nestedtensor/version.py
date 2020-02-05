__version__ = '0.0.1.dev20202423+f6c96f3'
git_version = 'f6c96f38545761975c0998f012f02d8393dfc530'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
