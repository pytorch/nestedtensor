__version__ = '0.1.4+7ad73c1'
git_version = '7ad73c1940229d423a8224da41ae662ea5a031b8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
