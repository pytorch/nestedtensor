__version__ = '0.0.1.dev202052121+84943eb'
git_version = '84943eb03e41bfb82e6847e4660a0a29bc42d503'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
