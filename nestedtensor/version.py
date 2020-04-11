__version__ = '0.0.1.dev202041018+14819da'
git_version = '14819da098e6943f3de76527340ea0357ae7efdd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
