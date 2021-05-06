__version__ = '0.1.4+8ffedb6'
git_version = '8ffedb6f5025ba80176cdf4f2cce0b53a2ca7470'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
