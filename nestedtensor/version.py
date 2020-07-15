__version__ = '0.0.1.dev20207152+17fbe15'
git_version = '17fbe151b7d2107624e55ae4054975f7bf5857d5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
