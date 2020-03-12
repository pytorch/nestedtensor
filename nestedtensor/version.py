__version__ = '0.0.1.dev20203122+19860a0'
git_version = '19860a0eb16b9d816969dc633b58b8b2db6afaca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
