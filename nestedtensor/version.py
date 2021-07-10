__version__ = '0.1.4+b0557d7'
git_version = 'b0557d7c694e1406c6a0a6cff323adf03200b4df'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
