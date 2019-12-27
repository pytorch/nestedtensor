__version__ = '0.0.1.dev201912272+a0bd8a8'
git_version = 'a0bd8a8e91f4c0d616f177dbd4376a747610321a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
