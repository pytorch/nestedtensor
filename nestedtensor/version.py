__version__ = '0.0.1.dev202052119+40b9b74'
git_version = '40b9b74f449da176d70b9302a2dfcd2db52e4c1e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
