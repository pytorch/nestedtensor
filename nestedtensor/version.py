__version__ = '0.0.1.dev2019121916+30be2c7'
git_version = '30be2c7a467e3d9ebf36d98fa252485a2d061e4d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
