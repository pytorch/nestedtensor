__version__ = '0.0.1.dev202013117+f3dfe30'
git_version = 'f3dfe3063d64e939cc1bdde68e4e8a71a7922f2f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
