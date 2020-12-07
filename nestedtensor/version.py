__version__ = '0.0.1+74e6a00'
git_version = '74e6a00a24b8f5097b0b9bca46c130f0a687f059'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
