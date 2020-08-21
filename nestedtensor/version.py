__version__ = '0.0.1.dev202082115+44b3da3'
git_version = '44b3da328960de006a7120d9a631bb6741411f57'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
