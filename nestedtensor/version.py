__version__ = '0.0.1.dev20206171+ee917ca'
git_version = 'ee917ca9efc161a7f79116acd010c41998dc9761'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
