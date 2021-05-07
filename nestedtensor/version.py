__version__ = '0.1.4+0bf6954'
git_version = '0bf6954c77bc5503b10838ec5aeb577054eaae52'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
