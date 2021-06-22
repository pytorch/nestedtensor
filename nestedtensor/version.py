__version__ = '0.1.4+50ca330'
git_version = '50ca33050448d2754e817d8b4641ee88cf4e8961'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
