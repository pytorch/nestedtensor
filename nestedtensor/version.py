__version__ = '0.0.1.dev20201919+d8734b8'
git_version = 'd8734b84bcdd5dd1c74b2c1f48f8c890c783925a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
