__version__ = '0.0.1.dev202082817+39521c7'
git_version = '39521c7daaa7c89e906c6092d1407144a31a290a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
