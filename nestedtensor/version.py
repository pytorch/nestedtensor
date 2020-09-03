__version__ = '0.0.1.dev2020933+f9e8024'
git_version = 'f9e8024d229128f12583d54d66d285bc5c78078e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
