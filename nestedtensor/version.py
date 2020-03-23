__version__ = '0.0.1.dev202032317+477255b'
git_version = '477255baa682353a212688f43248e49a01aea529'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
