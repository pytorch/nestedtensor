__version__ = '0.0.1.dev202011172+38087d0'
git_version = '38087d08a7101bd32488ad0f584a5eb027bfa27f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
