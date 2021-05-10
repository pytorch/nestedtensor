__version__ = '0.1.4+6f7030e'
git_version = '6f7030e6e7f7e2d4acc84ca0c41dbc4d1d866a65'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
