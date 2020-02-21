__version__ = '0.0.1.dev202022118+0823bce'
git_version = '0823bce5ca0d44fbcc7e0fb1b69a24c41e69eb86'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
