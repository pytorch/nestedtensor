__version__ = '0.0.1.dev202071519+bff2ba7'
git_version = 'bff2ba75d2d1c1139d1d35a19dc450c9b652ae7f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
