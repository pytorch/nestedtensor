__version__ = '0.0.1.dev20208255+faee8a1'
git_version = 'faee8a1a2578f7ecb80098d2cb792ea7c22e61ab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
