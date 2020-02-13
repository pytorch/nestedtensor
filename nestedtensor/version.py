__version__ = '0.0.1.dev20202136+f89a18a'
git_version = 'f89a18aa9521e1f2cf6305a4f41775e979b89861'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
