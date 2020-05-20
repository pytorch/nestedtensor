__version__ = '0.0.1.dev20205203+20c91be'
git_version = '20c91bec92e22e5df3ebbb48ad8c19be8f7477d5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
