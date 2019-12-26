__version__ = '0.0.1.dev2019122616+9722e9c'
git_version = '9722e9caec270e3b1b5494f49cc7e087f953690a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
