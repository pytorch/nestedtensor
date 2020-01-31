__version__ = '0.0.1.dev202013116+158bc31'
git_version = '158bc318f22b28346afc1217771750843d45c72b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
