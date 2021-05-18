__version__ = '0.1.4+06339d4'
git_version = '06339d4a36d75cf78c60fd1c90fa167149304620'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
