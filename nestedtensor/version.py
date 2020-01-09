__version__ = '0.0.1.dev20201918+e3decf5'
git_version = 'e3decf5a1b7d6d67e745e8fb07e4a4d749023732'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
