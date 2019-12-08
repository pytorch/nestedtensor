__version__ = '0.0.1.dev20191280+7f607fc'
git_version = '7f607fca6204840f070bd74cdc1a8f3099400eb1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
