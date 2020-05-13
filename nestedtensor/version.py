__version__ = '0.0.1.dev202051322+747306b'
git_version = '747306bc64c099628d10a20b0cc5090ab5c53328'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
