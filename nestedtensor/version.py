__version__ = '0.0.1.dev20208154+5ef0bd5'
git_version = '5ef0bd51ce1f79378d6b5a65b2fa9e4441f4f8a9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
