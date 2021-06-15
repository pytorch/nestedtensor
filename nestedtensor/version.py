__version__ = '0.1.4+bec2589'
git_version = 'bec2589724784730753a6a9a52f331884740b007'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
