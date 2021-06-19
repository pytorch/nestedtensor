__version__ = '0.1.4+5e3743a'
git_version = '5e3743aa9bcc2b230fa7318f4e4920c974086836'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
