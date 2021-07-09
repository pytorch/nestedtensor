__version__ = '0.1.4+67c8a73'
git_version = '67c8a73c4a53a24901a1c176e60729350c6dd6f1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
