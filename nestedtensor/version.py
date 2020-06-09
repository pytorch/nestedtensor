__version__ = '0.0.1.dev20206919+18ccf04'
git_version = '18ccf049d38d3ae112505e284d0c59d89d479250'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
