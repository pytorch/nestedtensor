__version__ = '0.0.1.dev202011221+44edac5'
git_version = '44edac542739f43b1375d1066668d7300a18e58b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
