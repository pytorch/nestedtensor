__version__ = '0.0.1+72fc7d4'
git_version = '72fc7d4459263b24b115453593ca5da0880dbce0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
