__version__ = '0.1.4+7958211'
git_version = '7958211d505470ac5e21d9c81e6db7fc5862a29e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
