__version__ = '0.0.1.dev201912262+a7d9095'
git_version = 'a7d9095b88155c245af3e9a16d1f8a88b0df87d7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
