__version__ = '0.0.1.dev20206171+393432c'
git_version = '393432ce37d7070d8d4ed35b2e68b39e6f23e7f4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
