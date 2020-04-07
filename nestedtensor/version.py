__version__ = '0.0.1.dev20204719+b778e2f'
git_version = 'b778e2f3c226a10fb05b0e0e4d0b374f95e4c1ef'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
