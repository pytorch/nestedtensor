__version__ = '0.0.1.dev202071515+2fb94d8'
git_version = '2fb94d8d788650f4bf4340988b0a2c0a3684fbe2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
