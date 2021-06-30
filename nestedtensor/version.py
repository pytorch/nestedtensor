__version__ = '0.1.4+e56fbdb'
git_version = 'e56fbdb0ae2974fdf8b3d4f1c5648ffc86de487e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
