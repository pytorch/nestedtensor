__version__ = '0.0.1.dev202083019+0061946'
git_version = '0061946186387cbcb95ae79a5c60f41c73ff1d17'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
