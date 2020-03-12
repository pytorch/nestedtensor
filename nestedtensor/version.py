__version__ = '0.0.1.dev20203126+f8d1d70'
git_version = 'f8d1d70bae446d3918c58b0960a584b7793506a6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
