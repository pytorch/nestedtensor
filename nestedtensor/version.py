__version__ = '0.1.4+952c721'
git_version = '952c7214bb8c94370e6b38a2d7a9371a0ca828d9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
