__version__ = '0.0.1.dev20205162+a6d55e0'
git_version = 'a6d55e0a5fa9f05387469ffd68242663612e1061'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
