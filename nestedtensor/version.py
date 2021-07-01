__version__ = '0.1.4+d7b7c02'
git_version = 'd7b7c02cca1d6a1181bfb0069934d7db40335cdc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
