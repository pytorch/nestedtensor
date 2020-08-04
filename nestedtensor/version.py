__version__ = '0.0.1.dev20208414+d11491b'
git_version = 'd11491b99de7bd8393b3a225679fa991c5cc2bb5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
