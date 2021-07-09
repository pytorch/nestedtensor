__version__ = '0.1.4+6664978'
git_version = '6664978e09b4259711f1dd6de8a3aea7b51ced13'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
