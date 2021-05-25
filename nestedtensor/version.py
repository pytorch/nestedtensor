__version__ = '0.1.4+885e7b8'
git_version = '885e7b8cbff40306fd20d5b7a314ffc6fa442079'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
