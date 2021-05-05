__version__ = '0.1.4+c691938'
git_version = 'c691938b1558b1768d1d7cd3c3a98d4152acc101'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
