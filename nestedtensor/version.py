__version__ = '0.0.1.dev20207184+97b874c'
git_version = '97b874c6d63e29998ea3d8c777d554f6a16157e3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
