__version__ = '0.0.1.dev20201918+ca3d2d5'
git_version = 'ca3d2d55e68df46427ec783445a79efd17c7fd3e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
