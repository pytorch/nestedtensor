__version__ = '0.1.4+db00bc4'
git_version = 'db00bc4c99fb1726094ab4804aaf816dff75a982'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
