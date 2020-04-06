__version__ = '0.0.1.dev20204621+cbdf67a'
git_version = 'cbdf67a92a9c20e5752598249fe43c5a55d6cb55'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
