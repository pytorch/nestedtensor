__version__ = '0.1.4+3e535a7'
git_version = '3e535a7ff0b8c12126fbb202eec9943b7e7ee09d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
