__version__ = '0.0.1.dev202082119+ddfb3cb'
git_version = 'ddfb3cb53a16386fc62c5116865d5eb4cbe6c240'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
