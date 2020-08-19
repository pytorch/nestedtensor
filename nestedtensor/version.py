__version__ = '0.0.1.dev20208194+d1b61d0'
git_version = 'd1b61d081614f461a36647f0a22239297c78977f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
