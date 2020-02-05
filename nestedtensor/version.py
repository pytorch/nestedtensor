__version__ = '0.0.1.dev2020250+28faef2'
git_version = '28faef2da22be9a4643067e21b2ac63580156d0e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
