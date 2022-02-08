__version__ = '0.1.4+62b77ac'
git_version = '62b77ace4a6ed5fbb67a6f60f20120883833ca4d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
