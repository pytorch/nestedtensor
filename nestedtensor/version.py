__version__ = '0.0.1.dev20209819+2379708'
git_version = '237970860878631099d6372d1dbc6628bbeeca37'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
