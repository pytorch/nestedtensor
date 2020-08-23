__version__ = '0.0.1.dev202082322+db55c47'
git_version = 'db55c47820873e39cfced503229e3d539a17f35f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
