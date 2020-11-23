__version__ = '0.0.1.dev2020112318+970350e'
git_version = '970350e0a8c03c5d5d60aede7713028f46ed2da9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
