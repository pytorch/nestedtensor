__version__ = '0.0.1.dev202032118+c4488fd'
git_version = 'c4488fdc5b9912747575c60611661b2217365e75'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
