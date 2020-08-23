__version__ = '0.0.1.dev202082322+6768b35'
git_version = '6768b35d4f09a4f8923b555c52d519b16d885c72'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
