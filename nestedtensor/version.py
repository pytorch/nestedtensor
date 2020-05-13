__version__ = '0.0.1.dev20205131+7e7d323'
git_version = '7e7d323dfa659b2f1f701a5b07b721a1dc141ad4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
