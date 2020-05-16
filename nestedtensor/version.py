__version__ = '0.0.1.dev20205161+cc800d0'
git_version = 'cc800d091115c33f0b8f84d7c5c4bedfe3f73c85'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
