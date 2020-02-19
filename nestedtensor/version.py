__version__ = '0.0.1.dev202021919+db8eb98'
git_version = 'db8eb98bc4a7f0277042458b537c7ca4256b3d2f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
