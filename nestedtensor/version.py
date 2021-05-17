__version__ = '0.1.4+11f3d46'
git_version = '11f3d46cf4426c2e670bfa98fb3b6ada1655d808'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
