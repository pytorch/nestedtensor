__version__ = '0.1.4+c18d6dc'
git_version = 'c18d6dc1f3a49c79ee6a8d01dd2cf4cc793792e4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
