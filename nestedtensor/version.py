__version__ = '0.0.1.dev20208619+41574b5'
git_version = '41574b58e752f04c47ca8dedb05a2d2cc4da9fe4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
