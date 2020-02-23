__version__ = '0.0.1.dev20202231+14ea163'
git_version = '14ea1633a5c2a193b6d3e0e6382486a0a970ff51'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
