__version__ = '0.0.1+28db5c4'
git_version = '28db5c433e92f0d052d191475ee88bdaf3b02ca6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
