__version__ = '0.0.1.dev20201202+d712aab'
git_version = 'd712aab7257df7781d0b9f8f2c2643f5a26e9108'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
