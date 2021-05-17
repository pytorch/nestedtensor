__version__ = '0.1.4+2a58d05'
git_version = '2a58d0548088c20e5952f91ca03fa0d3251f07c5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
