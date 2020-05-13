__version__ = '0.0.1.dev20205131+d14fc69'
git_version = 'd14fc69d4b293e06f987a9bf3d8ac25b9ea9fcc9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
