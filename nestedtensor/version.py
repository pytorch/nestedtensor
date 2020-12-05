__version__ = '0.0.1+98505fa'
git_version = '98505fa49f045577afd3d47fd44293badd5b6ab9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
